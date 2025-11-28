#!/usr/bin/env python3
#
# File   : nerf.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import functools
from typing import Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple

import gin
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import jax

from dycheck import geometry
from dycheck.nn import Embed, NeRFMLP, PosEnc, RayWarping, PoseGraph_Blurry, LocalObjectMotionMLP, Pluecker
from dycheck.nn import functional as F
from dycheck.utils import common, struct, types



@gin.configurable(denylist=["name"])
class MoBluRF(nn.Module):
    # Data specifics.
    embeddings_dict: Mapping[
        Literal["time", "camera"], Sequence[int]
    ] = gin.REQUIRED
    near: float = gin.REQUIRED
    far: float = gin.REQUIRED
    num_frames: int = gin.REQUIRED 
    scene_origin: jnp.ndarray = gin.REQUIRED

    # Architecture.
    use_warp: bool = False
    points_embed_key: Literal["time"] = "time"
    points_embed_cls: Callable[..., nn.Module] = functools.partial(
        Embed,
        features=8,
    )
    rgb_embed_key: Optional[Literal["time", "camera"]] = None
    rgb_embed_cls: Callable[..., nn.Module] = functools.partial(
        Embed,
        features=8,
    )
    use_viewdirs: bool = False
    viewdirs_embed_cls: Callable[..., nn.Module] = functools.partial(
        PosEnc,
        num_freqs=4,
        use_identity=True,
    )
    sigma_activation: types.Activation = nn.softplus

    # Stage.
    stage: str = 'bri'

    # Rendering.
    num_coarse_samples: int = 128
    num_fine_samples: int = 128
    use_randomized: bool = True
    noise_std: Optional[float] = None
    use_white_bkgd: bool = False
    use_linear_disparity: bool = False
    use_sample_at_infinity: bool = True
    use_cull_cameras: bool = False
    correct_pose: bool = True
    cameras_dict: Optional[
        Dict[
            Literal[
                "intrin",
                "extrin",
                "c2w",
                "radial_distortion",
                "tangential_distortion",
                "image_size",
            ],
            jnp.ndarray,
        ]
    ] = None
    num_min_frames: int = 5
    min_frame_ratio: float = 0.1

    # Biases.
    resample_padding: float = 0.0
    sigma_bias: float = 0.0
    rgb_padding: float = 0.0

    # Evaluation.
    # On-demand exclusion to save memory.
    exclude_fields: Tuple[str] = ()
    # On-demand returning to save memory. Will override exclusion.
    return_fields: Tuple[str] = ()

    @property
    def use_fine(self) -> bool:
        return self.num_fine_samples > 0

    @property
    def num_points_embeds(self):
        return max(self.embeddings_dict[self.points_embed_key]) + 1

    @property
    def use_rgb_embed(self) -> bool:
        return self.rgb_embed_key is not None

    @property
    def num_rgb_embeds(self):
        return max(self.embeddings_dict[self.rgb_embed_key]) + 1

    def setup(self):
        points_embed_cls = common.tolerant_partial(
            self.points_embed_cls, num_embeddings=self.num_points_embeds
        )
        self.points_embed = points_embed_cls()

        if self.use_rgb_embed:
            rgb_embed_cls = common.tolerant_partial(
                self.rgb_embed_cls, num_embeddings=self.num_rgb_embeds
            )
            self.rgb_embed = rgb_embed_cls()

        if self.use_viewdirs:
            self.viewdirs_embed = self.viewdirs_embed_cls()

        nerfs = {
            "coarse_net": NeRFMLP(),
            "static_net": NeRFMLP(pred_blending = True),
            "dynamic_net": NeRFMLP()
        }

        # pose params
        self.ray_warping = RayWarping(self.num_frames)
        self.nerfs = nerfs
        
        if self.stage == 'mdd':
            self.pose_blurry = PoseGraph_Blurry(self.num_frames)
            self.blurry_step = self.pose_blurry.blurry_step

            # for LORR
            self.local_object_motion_mlp = LocalObjectMotionMLP(num_frames = self.num_frames)
            self.ray_embed = Pluecker(origin=self.scene_origin)

        # grid sample
        def interpolation_fn(
            x, idx): return x[idx[Ellipsis, 1], idx[Ellipsis, 0]]
        self.vmap_interpolation_fn = jax.jit(
            jax.vmap(interpolation_fn, (0, 0), 0))

    def get_conditions(
        self, samples: struct.Samples
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        trunk_conditions, rgb_conditions = [], []

        if self.rgb_embed_key is not None:
            assert samples.metadata is not None
            rgb_embed = getattr(samples.metadata, self.rgb_embed_key)
            rgb_embed = self.rgb_embed(rgb_embed)
            rgb_conditions.append(rgb_embed)

        if self.use_viewdirs:
            viewdirs_embed = self.viewdirs_embed(samples.directions)
            rgb_conditions.append(viewdirs_embed)

        trunk_conditions = (
            jnp.concatenate(trunk_conditions, axis=-1)
            if trunk_conditions
            else None
        )
        rgb_conditions = (
            jnp.concatenate(rgb_conditions, axis=-1)
            if rgb_conditions
            else None
        )
        return trunk_conditions, rgb_conditions

    def embed_samples(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
        only_spatial: bool = False,
    ):
        if use_warp is None:
            use_warp = self.use_warp
        else:
            assert self.use_warp, "The model does not support warping."
        if use_warp:
            warp_out = self.points_embed(
                samples, extra_params, return_jacobian=use_warp_jacobian
            )
            assert "warped_points_embed" in warp_out
            warped_points_embed = warp_out.pop("warped_points_embed")
        else:
            # Return original points if no warp has been applied.
            warp_out = {"warped_points": samples.xs}
            if self.use_warp:
                warped_points_embed = self.points_embed.warped_points_embed(
                    xs=samples.xs
                )
            elif only_spatial: # embed xyz
                warped_points_embed = self.points_embed(
                    xs=samples.xs,
                    metadata=None,
                )
            else: # embed xyzt
                warped_points_embed = self.points_embed(
                    xs=samples.xs,
                    metadata=getattr(samples.metadata, self.points_embed_key),
                    w_time=extra_params.w_time,
                )
        return warped_points_embed, warp_out

    def eval_samples(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
        level: Optional[Literal["coarse_net", "static_net", "dynamic_net"]] = None,
        use_randomized: Optional[bool] = None,
        exclude_fields: Optional[Tuple[str]] = None,
        return_fields: Optional[Tuple[str]] = None,
        protect_fields: Tuple[str] = (),
        is_training: Optional[bool] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Evaluate points at given positions.

        Assumes (N, S, 3) points, (N, S, 3) viewdirs, (N, S, 1) metadata.

        Supported fields:
             - point_sigma
             - point_rgb
             - point_feat
             - points
             - warped_points
             - warp_out
        """
        if use_randomized is None:
            use_randomized = self.use_randomized
        if exclude_fields is None:
            exclude_fields = self.exclude_fields
        if return_fields is None:
            return_fields = self.return_fields

        assert level is not None, "Level must be explicitly provided."
        nerf = self.nerfs[level]
        
        trunk_conditions, rgb_conditions = self.get_conditions(samples)

        out = {"points": samples.xs}

        warped_points_embed, warp_out = self.embed_samples(
            samples,
            extra_params,
            use_warp=use_warp,
            use_warp_jacobian=use_warp_jacobian,
            only_spatial = True if level == "static_net" else False
        )
        out["warp_out"] = warp_out

        if level == "static_net":
            return_fields = (
                "point_sigma",
                "point_rgb",
                "point_blending"
            )
        else:
            return_fields = (
                "point_sigma",
                "point_rgb",
            )

        rng_level = "static_net" if level == "static_net" else "dynamic_net"
        logits = nerf(warped_points_embed, trunk_conditions, rgb_conditions, return_fields)
        logits = F.rendering.perturb_logits(
            self.make_rng(rng_level), logits, use_randomized, self.noise_std
        )

        # Apply activations.
        logits["point_sigma"] = self.sigma_activation(
            logits["point_sigma"] + self.sigma_bias
        )
        if "point_rgb" in logits:
            logits["point_rgb"] = (
                nn.sigmoid(logits["point_rgb"]) * (1 + 2 * self.rgb_padding)
                - self.rgb_padding
            )
        if "point_blending" in logits:
            logits["point_blending"] = nn.sigmoid(logits["point_blending"]) # apply sigmoid to blending

        out.update(logits)
        return out

    def render_samples_blending(self, out_static, out_dynamic, samples):

        bkgd_rgb = jnp.full(
            (3,), 1 if self.use_white_bkgd else 0, dtype=jnp.float32
        )

        blending_outputs = F.rendering.volrend_blending(
                out_static,
                out_dynamic,
                samples,
                bkgd_rgb=bkgd_rgb,
                use_sample_at_infinity=self.use_sample_at_infinity,
                stage=self.stage
            )

        return blending_outputs

    def render_samples(
        self,
        samples: struct.Samples,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
        level: Optional[str] = None,
        use_randomized: Optional[bool] = None,
        bkgd_rgb: Optional[jnp.ndarray] = None,
        use_cull_cameras: Optional[bool] = None,
        exclude_fields: Optional[str] = None,
        return_fields: Optional[str] = None,
        protect_fields: Tuple[str] = (),
        rays:  Optional[struct.Rays] = None,
        is_training: Optional[bool] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Note that tvals is of shape (N, S + 1) such that it covers the start
        and end of the ray. Samples are evaluated at mid points.

        Supported fields:
            - point_sigma
            - point_rgb
            - point_feat
            - points
            - warped_points
            - warp_out
            - rgb
            - depth
            - med_depth
            - acc
            - alpha
            - trans
            - weights
        """
        if use_cull_cameras is None:
            use_cull_cameras = self.use_cull_cameras
        if exclude_fields is None:
            exclude_fields = self.exclude_fields
        if return_fields is None:
            return_fields = self.return_fields

        # Return all fields and filter after.
        out = self.eval_samples(
            samples,
            extra_params,
            use_warp=use_warp,
            use_warp_jacobian=use_warp_jacobian,
            level=level,
            use_randomized=use_randomized,
            is_training=is_training,
        )

        if bkgd_rgb is None:
            bkgd_rgb = jnp.full(
                (3,), 1 if self.use_white_bkgd else 0, dtype=jnp.float32
            )
        out.update(
            F.rendering.volrend(
                out,
                samples,
                bkgd_rgb=bkgd_rgb,
                use_sample_at_infinity=self.use_sample_at_infinity,
            )
        )

        return out

    def correct_pose_fn(self, rays):
        time_index = rays.metadata.time.squeeze(-1) # B
        N_rays = time_index.shape[0]
        intrins = jnp.take_along_axis(self.cameras_dict["intrin"], jnp.broadcast_to(time_index[:,None,None], (N_rays,3,3)), axis=0)
        w2cs = jnp.take_along_axis(self.cameras_dict["extrin"][:,:3,:4], jnp.broadcast_to(time_index[:,None,None], (N_rays,3,4)), axis=0)

        pixels = rays.pixels

        Rw2c_new, tw2c_new = self.ray_warping(w2cs, time_index)
        Rc2w_new = jnp.transpose(Rw2c_new, (0,2,1))
        
        directions_new, _ = geometry.get_rays_direction(pixels, intrins, Rc2w_new)
        origins_new = -Rc2w_new@(tw2c_new[...,None])
        
        new_rays = struct.Rays(
            origins=origins_new.squeeze(-1),
            directions=directions_new,
            pixels=pixels,
            metadata=rays.metadata
        )

        if self.stage == 'bri':
            neighbor_rays = self.calculate_neighbor_rays(new_rays, intrins, Rc2w_new)
            return new_rays, neighbor_rays
        
        elif self.stage == 'mdd':
            w2cs_new = jnp.concatenate((Rw2c_new, tw2c_new[...,None]), axis=-1)
            return new_rays, w2cs_new, time_index, pixels, intrins, Rc2w_new


    def calculate_neighbor_rays(self, new_rays, intrins, Rc2w_new):
        """Calculates rays for neighboring pixels for normal estimation."""
        num_neighbor = 4
        curr_origins = new_rays.origins
        curr_pixels = new_rays.pixels
        metadata_time = new_rays.metadata.time
        metadata_camera = new_rays.metadata.camera

        neighbor_origins = jnp.broadcast_to(curr_origins[:,None], curr_origins.shape[:1] + (num_neighbor,) + curr_origins.shape[1:])
        neighbor_origins = neighbor_origins.reshape((-1,) + neighbor_origins.shape[2:])

        neighbor_metadata_time = jnp.broadcast_to(metadata_time[:,None], metadata_time.shape[:1] + (num_neighbor,) + metadata_time.shape[1:])
        neighbor_metadata_time = neighbor_metadata_time.reshape((-1,) + neighbor_metadata_time.shape[2:])

        neighbor_metadata_camera = jnp.broadcast_to(metadata_camera[:,None], metadata_camera.shape[:1] + (num_neighbor,) + metadata_camera.shape[1:])
        neighbor_metadata_camera = neighbor_metadata_camera.reshape((-1,) + neighbor_metadata_camera.shape[2:])

        neighbor_metadata = struct.Metadata(
            time = neighbor_metadata_time,
            camera = neighbor_metadata_camera
        )
        
        img_w, img_h = self.cameras_dict["image_size"][0]
        neighbor_grids = jnp.array([
            [-1,0], [1,0], [0,-1], [0, 1] # left, right, up bottom
        ]).astype(jnp.float32) # 4, 2
        neighbor_grids = neighbor_grids[None]

        raw_neighbor_pixels = curr_pixels[:,None] + neighbor_grids #nrays, 4 ,2
        neighbor_pixels = jnp.stack([
            jnp.clip(raw_neighbor_pixels[...,0], 0.5, img_w - 0.5),
            jnp.clip(raw_neighbor_pixels[...,1], 0.5, img_h - 0.5)
        ], axis = -1) # clip edges

        neighbor_pixels = neighbor_pixels.reshape((-1,) + neighbor_pixels.shape[2:])
        intrins = jnp.broadcast_to(intrins[:,None], intrins.shape[:1] + (num_neighbor,) + intrins.shape[1:])
        intrins = intrins.reshape((-1,) + intrins.shape[2:])

        Rc2w_new = jnp.broadcast_to(Rc2w_new[:,None], Rc2w_new.shape[:1] + (num_neighbor,) + Rc2w_new.shape[1:])
        Rc2w_new = Rc2w_new.reshape((-1,) + Rc2w_new.shape[2:])

        neighbor_directions, raw_local_viewdirs = geometry.get_rays_direction(neighbor_pixels, intrins, Rc2w_new)

        neighbor_rays = struct.Rays(
            origins=neighbor_origins,
            directions=neighbor_directions,
            pixels=neighbor_pixels,
            local_directions=raw_local_viewdirs,
            metadata=neighbor_metadata,
        )
        
        return neighbor_rays

    def ilsp(self, new_rays, w2cs_new, time_index, pixels, intrins, Rc2w_new, motion_mask):
        # GCRP
        blurry_Rw2c_new, blurry_tw2c_new =  self.pose_blurry(w2cs_new, time_index)
        blurry_w2cs_new = jnp.concatenate((blurry_Rw2c_new, blurry_tw2c_new[...,None]), axis=-1)

        blurry_Rc2w_new = jnp.transpose(blurry_Rw2c_new, (0,2,1))

        blurry_intrins = jnp.broadcast_to(intrins[:,None], intrins.shape[:1] + (self.blurry_step,) + intrins.shape[1:])
        blurry_intrins = blurry_intrins.reshape((-1,) + blurry_intrins.shape[2:])

        blurry_pixels = jnp.broadcast_to(pixels[:,None], pixels.shape[:1] + (self.blurry_step,) + pixels.shape[1:])
        blurry_pixels = blurry_pixels.reshape((-1,) + blurry_pixels.shape[2:])

        blurry_directions_new, _ = geometry.get_rays_direction(blurry_pixels, blurry_intrins, blurry_Rc2w_new)
        blurry_origins_new = -blurry_Rc2w_new@(blurry_tw2c_new[...,None])

        metadata_time = new_rays.metadata.time
        metadata_camera = new_rays.metadata.camera

        blurry_metadata_time = jnp.broadcast_to(metadata_time[:,None], metadata_time.shape[:1] + (self.blurry_step,) + metadata_time.shape[1:])
        blurry_metadata_time = blurry_metadata_time.reshape((-1,) + blurry_metadata_time.shape[2:])

        blurry_metadata_camera = jnp.broadcast_to(metadata_camera[:,None], metadata_camera.shape[:1] + (self.blurry_step,) + metadata_camera.shape[1:])
        blurry_metadata_camera = blurry_metadata_camera.reshape((-1,) + blurry_metadata_camera.shape[2:])

        blurry_metadata = struct.Metadata(
            time = blurry_metadata_time,
            camera = blurry_metadata_camera
        )

        # LORR
        blurry_embed_rays = struct.Rays(
            origins=blurry_origins_new.squeeze(-1),
            directions=blurry_directions_new,
            pixels=blurry_pixels,
            metadata=blurry_metadata
        )

        embed_samples = F.sampling.uniform(
            self.make_rng("static_net"),
            blurry_embed_rays,
            32,
            self.near,
            self.far,
            use_randomized=False,
            use_linear_disparity=self.use_linear_disparity,
        )
        ray_embed = embed_samples.xs.reshape((embed_samples.xs.shape[0], -1))

        blurry_time_embed = self.points_embed.embed(blurry_metadata_time)
        expand_motion_mask = jnp.broadcast_to(motion_mask[:,None], (motion_mask.shape[0], self.blurry_step, 1)).reshape(-1,1)
        raytime_embed = jnp.concatenate([ray_embed, blurry_time_embed], axis=-1)
        motion_blurry_Rw2c_new, motion_blurry_tw2c_new = self.local_object_motion_mlp(raytime_embed, blurry_metadata_time.squeeze(-1), blurry_w2cs_new, expand_motion_mask)

        motion_blurry_Rc2w_new = jnp.transpose(motion_blurry_Rw2c_new, (0,2,1))
        motion_blurry_directions_new, _ = geometry.get_rays_direction(blurry_pixels, blurry_intrins, motion_blurry_Rc2w_new)
        motion_blurry_origins_new = -motion_blurry_Rc2w_new@(motion_blurry_tw2c_new[...,None])

        blurry_rays = struct.Rays(
            origins=motion_blurry_origins_new.squeeze(-1),
            directions=motion_blurry_directions_new,
            pixels=blurry_pixels,
            metadata=blurry_metadata
        )

        # create neighbor rays        
        neighbor_rays = self.calculate_neighbor_rays(new_rays, intrins, Rc2w_new)
        
        return blurry_rays, neighbor_rays

    def compute_blurry_offsets(self, samples, 
                            extra_params: Optional[struct.ExtraParams],
                            use_warp: Optional[bool] = None,
                            use_warp_jacobian: bool = False,):
        warped_points_embed, _ = self.embed_samples(
            samples,
            extra_params,
            use_warp=use_warp,
            use_warp_jacobian=use_warp_jacobian,
            only_spatial = False
        )

        points_offsets = self.offsets_mlp(warped_points_embed)
        return points_offsets

    def __call__(
        self,
        rays: struct.Rays,
        extra_params: Optional[struct.ExtraParams],
        use_warp: Optional[bool] = None,
        use_warp_jacobian: bool = False,
        use_randomized: Optional[bool] = None,
        bkgd_rgb: Optional[jnp.ndarray] = None,
        use_cull_cameras: Optional[bool] = None,
        correct_pose: Optional[bool] = None,
        exclude_fields: Optional[Tuple[str]] = None,
        return_fields: Optional[Tuple[str]] = None,
        protect_fields: Tuple[str] = (),
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
        """
        Supported fields:
            - coarse/fine
                - point_sigma
                - point_rgb
                - point_feat
                - points
                - warped_points
                - warp_out
                    - scores
                    - weights
                - rgb
                - depth
                - med_depth
                - acc
                - alpha
                - trans
                - weights
                - tvals
        """
        if use_randomized is None:
            use_randomized = self.use_randomized
        if exclude_fields is None:
            exclude_fields = self.exclude_fields
        if return_fields is None:
            return_fields = self.return_fields
        if extra_params is None:
            extra_params = struct.ExtraParams(
                warp_alpha=jnp.zeros((1,), jnp.float32),
                ambient_alpha=jnp.zeros((1,), jnp.float32),
                w_time=jnp.zeros((1,), jnp.float32),
                current_step=jnp.zeros((1,), jnp.uint32),
            )

        # Brute-force adding prefix to return_fields if there's any. It will be
        # used by the training loop.
        return_fields = (
            return_fields
            + tuple([f"coarse/{f}" for f in return_fields])  # type: ignore
            + tuple([f"fine/{f}" for f in return_fields])  # type: ignore
            + tuple([f"full/{f}" for f in return_fields])  # type: ignore
        )

        if correct_pose is None:
            correct_pose = self.correct_pose

        if correct_pose:
            if self.stage == 'bri':
                rays, neighbor_rays = self.correct_pose_fn(rays)
            elif self.stage == 'mdd':
                rays_base, w2cs_new_base, time_index_base, pixels_base, intrins_base, Rc2w_new_base = self.correct_pose_fn(rays)
                rays = rays_base
            else:
                raise ValueError(f"Unknown stage: {self.stage}. Supported stages are 'bri' and 'mdd'.")

        render_samples = functools.partial(
            self.render_samples,
            extra_params=extra_params,
            use_warp=use_warp,
            use_randomized=use_randomized,
            bkgd_rgb=bkgd_rgb,
            use_cull_cameras=use_cull_cameras,
            is_training=correct_pose,
        )

        # Sample coarse points and render rays.
        samples = F.sampling.uniform(
            self.make_rng("static_net"),
            rays,
            self.num_coarse_samples,
            self.near,
            self.far,
            use_randomized=use_randomized,
            use_linear_disparity=self.use_linear_disparity,
        )
        # coarse_uniform_samples = samples

        coarse_net_out = render_samples(
            samples=samples,
            use_warp_jacobian=use_warp_jacobian,
            level="coarse_net",
        )

        out = {"coarse_net": coarse_net_out}
        out["coarse_net"]["tvals"] = samples.tvals

        if self.use_fine:
            assert samples.tvals is not None
            samples = F.sampling.ipdf(
                self.make_rng("dynamic_net"),
                0.5 * (samples.tvals[..., 1:, 0] + samples.tvals[..., :-1, 0]),
                out["coarse_net"]["weights"][..., 1:-1, 0],
                rays,
                samples,
                self.num_fine_samples,
                use_randomized=use_randomized,
            )
            out["dynamic_net"] = render_samples(
                samples=samples,
                use_warp_jacobian=use_warp_jacobian,
                level="dynamic_net",
            )
            out["dynamic_net"]["tvals"] = samples.tvals

        out["static_net"] = render_samples(
            samples=samples,
            level="static_net",
            protect_fields=("weights",) if self.use_fine else (),
            rays=rays,
        )
        out["static_net"]["tvals"] = samples.tvals

        # render blending
        render_full = self.render_samples_blending(
            out["static_net"],
            out["dynamic_net"],
            samples=samples,
        )

        out["full"] = render_full
        out["full"]["tvals"] = samples.tvals
        out["coarse_net"]["mask"] = out["full"]["mask"]
        out["static_net"]["mask"] = out["full"]["mask"]
        out["dynamic_net"]["mask"] = out["full"]["mask"]

        if self.stage == 'mdd':
            if correct_pose:
                blurry_rays, neighbor_rays = self.ilsp(rays_base, w2cs_new_base, time_index_base, pixels_base, intrins_base, Rc2w_new_base, out["full"]["mask"])

            # predict blurry rays 
            if correct_pose:
                blurry_samples = F.sampling.uniform(
                    self.make_rng("static_net"),
                    blurry_rays,
                    self.num_coarse_samples,
                    self.near,
                    self.far,
                    use_randomized=use_randomized,
                    use_linear_disparity=self.use_linear_disparity,
                )

                blurry_coarse_net_out = render_samples(
                    samples=blurry_samples,
                    use_warp_jacobian=use_warp_jacobian,
                    level="coarse_net",
                )

                if self.use_fine:
                    blurry_samples = F.sampling.ipdf(
                        self.make_rng("dynamic_net"),
                        0.5 * (blurry_samples.tvals[..., 1:, 0] + blurry_samples.tvals[..., :-1, 0]),
                        blurry_coarse_net_out["weights"][..., 1:-1, 0],
                        blurry_rays,
                        blurry_samples,
                        self.num_fine_samples,
                        use_randomized=use_randomized,
                    )

                    blurry_dynamic_net_out = render_samples(
                        samples=blurry_samples,
                        use_warp_jacobian=use_warp_jacobian,
                        level="dynamic_net",
                    )

                    blurry_static_net_out = render_samples(
                            samples=blurry_samples,
                            level="static_net",
                            protect_fields=("weights",) if self.use_fine else (),
                            rays=blurry_rays,
                        )
                    
                    blurry_render_full = self.render_samples_blending(
                        (blurry_static_net_out),
                        blurry_dynamic_net_out,
                        samples=samples,
                    )
                    
                    blurry_coarse_net_out = blurry_coarse_net_out['rgb'].reshape(-1, self.blurry_step, 3)
                    blurry_static_net_out = blurry_static_net_out['rgb'].reshape(-1, self.blurry_step, 3)
                    blurry_dynamic_net_out = blurry_dynamic_net_out['rgb'].reshape(-1, self.blurry_step, 3)
                    blurry_render_full = blurry_render_full['rgb'].reshape(-1, self.blurry_step, 3)

                out["coarse_net"]['rgb'] = jnp.concatenate([out['coarse_net']['rgb'][:,None], blurry_coarse_net_out], axis =1).mean(axis =1)
                out["static_net"]['rgb'] = jnp.concatenate([out['static_net']['rgb'][:,None], blurry_static_net_out], axis =1).mean(axis =1)
                out["dynamic_net"]['rgb'] = jnp.concatenate([out['dynamic_net']['rgb'][:,None], blurry_dynamic_net_out], axis =1).mean(axis =1)                
                out["full"]['rgb'] = jnp.concatenate([out['full']['rgb'][:,None], blurry_render_full], axis =1).mean(axis =1)

        if correct_pose:
            neighbor_samples = F.sampling.uniform(
                self.make_rng("static_net"),
                neighbor_rays,
                self.num_coarse_samples,
                self.near,
                self.far,
                use_randomized=use_randomized,
                use_linear_disparity=self.use_linear_disparity,
            )

            neighbor_coarse_net_out = render_samples(
                samples=neighbor_samples,
                use_warp_jacobian=use_warp_jacobian,
                level="coarse_net",
            )

            # compute normals
            z = neighbor_coarse_net_out["depth"]
            local_coords = neighbor_rays.local_directions * z 
            local_coords = local_coords.reshape(-1,4,3) # l,r,u,b order

            dxdu = (local_coords[:,1,0] - local_coords[:,0,0])[...,None] #rx-lx
            dydu = (local_coords[:,1,1] - local_coords[:,0,1])[...,None] #ry-ly
            dzdu = (local_coords[:,1,2] - local_coords[:,0,2])[...,None] #rz-lz

            dxdv = (local_coords[:,-1,0] - local_coords[:,-2,0])[...,None] #bx-ux
            dydv = (local_coords[:,-1,1] - local_coords[:,-2,1])[...,None] #by-uy
            dzdv = (local_coords[:,-1,2] - local_coords[:,-2,2])[...,None] #bz-uz

            n_x = dydv * dzdu - dydu * dzdv
            n_y = dzdv * dxdu - dzdu * dxdv
            n_z = dxdv * dydu - dxdu * dydv

            n = jnp.concatenate([n_x, n_y, n_z], axis=-1)
            normal_vec = (n / jnp.linalg.norm(n, axis=-1)[:,None])
            out["coarse_net"]["normal"] = normal_vec

            if self.use_fine:
                neighbor_samples = F.sampling.ipdf(
                    self.make_rng("dynamic_net"),
                    0.5 * (neighbor_samples.tvals[..., 1:, 0] + neighbor_samples.tvals[..., :-1, 0]),
                    neighbor_coarse_net_out["weights"][..., 1:-1, 0],
                    neighbor_rays,
                    neighbor_samples,
                    self.num_fine_samples,
                    use_randomized=use_randomized,
                )
                neighbor_dynamic_net_out = render_samples(
                    samples=neighbor_samples,
                    use_warp_jacobian=use_warp_jacobian,
                    level="dynamic_net",
                )

                # compute normals
                z = neighbor_dynamic_net_out["depth"]
                local_coords = neighbor_rays.local_directions * z 
                local_coords = local_coords.reshape(-1,4,3) # l,r,u,b order

                dxdu = (local_coords[:,1,0] - local_coords[:,0,0])[...,None] #rx-lx
                dydu = (local_coords[:,1,1] - local_coords[:,0,1])[...,None] #ry-ly
                dzdu = (local_coords[:,1,2] - local_coords[:,0,2])[...,None] #rz-lz

                dxdv = (local_coords[:,-1,0] - local_coords[:,-2,0])[...,None] #bx-ux
                dydv = (local_coords[:,-1,1] - local_coords[:,-2,1])[...,None] #by-uy
                dzdv = (local_coords[:,-1,2] - local_coords[:,-2,2])[...,None] #bz-uz

                n_x = dydv * dzdu - dydu * dzdv
                n_y = dzdv * dxdu - dzdu * dxdv
                n_z = dxdv * dydu - dxdu * dydv

                n = jnp.concatenate([n_x, n_y, n_z], axis=-1)
                normal_vec = (n / jnp.linalg.norm(n, axis=-1)[:,None])
                out["dynamic_net"]["normal"] = normal_vec

        # hardcode for return fields
        levels = ["coarse_net", "static_net", "dynamic_net", "full"]
        for level in levels:
            out[level] = common.traverse_filter(
                out[level],
                exclude_fields=(),
                return_fields=('weights', 'depth', 'rgb', 'tvals', 'mask', 'point_blending', "points", "ref_rgb", "ref_mask", "normal", "mask_smoothx", "mask_smoothy", 'raw_mask'),
                protect_fields=(),
                inplace=True,
            )

        return out

    @classmethod
    def create(
        cls,
        key: types.PRNGKey,
        embeddings_dict: Dict[Literal["time", "camera"], Sequence[int]],
        near: float,
        far: float,
        num_frames: int, 
        scene_origin: jnp.array,
        cameras_dict: Optional[
            Dict[
                Literal[
                    "intrin",
                    "extrin",
                    "c2w",
                    "radial_distortion",
                    "tangential_distortion",
                    "image_size",
                ],
                jnp.ndarray,
            ]
        ] = None,
        exclude_fields: Tuple[str] = (),
        return_fields: Tuple[str] = (),
    ):
        """Neural Randiance Field.

        Args:
            key (PRNGKey): PRNG key.
            embeddings_dict (Dict[str, Sequence[int]]): Dictionary of unique
                embeddings.
            near (float): Near plane.
            far (float): Far plane.
            exclude_fields (Tuple[str]): Fields to exclude.
            return_fields (Tuple[str]): Fields to return.

        Returns:
            model (nn.Model): the dynamic NeRF model.
            params (Dict[str, jnp.ndarray]): the parameters for the model.
        """

        model = cls(
            embeddings_dict,
            near=near,
            far=far,
            num_frames=num_frames,
            scene_origin=scene_origin,
            cameras_dict=cameras_dict,
            exclude_fields=exclude_fields,
            return_fields=return_fields,
        )

        rays = struct.Rays(
            origins=jnp.ones((1, 3), jnp.float32),
            directions=jnp.ones((1, 3), jnp.float32),
            pixels=jnp.ones((1, 2), jnp.float32),
            metadata=struct.Metadata(
                time=jnp.ones((1, 1), jnp.uint32),
                camera=jnp.ones((1, 1), jnp.uint32),
            ),
        )
        
        extra_params = struct.ExtraParams(
            warp_alpha=jnp.zeros((1,), jnp.float32),
            ambient_alpha=jnp.zeros((1,), jnp.float32),
            w_time=jnp.zeros((1,), jnp.float32),
            current_step=jnp.zeros((1,), jnp.uint32),
        )

        key, key0, key1, key2 = random.split(key, 4)
        variables = model.init(
            {"params": key, "coarse_net": key0, "static_net": key1, "dynamic_net": key2},
            rays=rays,
            extra_params=extra_params,
        )
        # create pose variables

        return model, variables
