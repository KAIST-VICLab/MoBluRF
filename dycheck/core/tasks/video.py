#!/usr/bin/env python3
#
# File   : novel_view.py
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

import dataclasses
import itertools
import os.path as osp
from collections import defaultdict
from typing import Dict, Literal, Optional, Sequence, Union

import gin
import numpy as np
from absl import logging

from dycheck import geometry
from dycheck.utils import common, io, struct, types, image

from . import base
from .functional import get_prender_image
import jax


@dataclasses.dataclass
class VideoConfig(object):
    camera_traj: Literal["fixed", "arc", "lemniscate"]
    time_traj: Literal["fixed", "replay"]
    camera_idx: int = 0
    time_idx: int = 0
    camera_traj_params: Dict = dataclasses.field(
        default_factory=lambda: {"num_frames": 60, "degree": 5}
    )
    fps: float = 20

    def __post_init__(self):
        assert not (self.camera_traj == "fixed" and self.time_traj == "fixed")

    def __repr__(self):
        if self.camera_traj == "fixed":
            return "Stabilized-view video"
        elif self.time_traj == "fixed":
            return "Novel-view video"
        else:
            return "Bullet-time video"

    @property
    def short_name(self):
        fps = float(self.fps)
        if self.camera_traj == "fixed":
            return (
                f"stabilized_view@ci={self.camera_idx}-ti={self.time_idx}-"
                f"fps={fps}"
            )
        elif self.time_traj == "fixed":
            cparams_str = "-".join(
                [f"{k}={v}" for k, v in self.camera_traj_params.items()]
            )
            return (
                f"novel_view@ci={self.camera_idx}-ti={self.time_idx}-"
                f"fps={fps}-ctraj={self.camera_traj}-{cparams_str}"
            )
        else:
            cparams_str = "-".join(
                [f"{k}={v}" for k, v in self.camera_traj_params.items()]
            )
            return (
                f"bullet_time@ci={self.camera_idx}-ti={self.time_idx}-"
                f"fps={fps}-ctraj={self.camera_traj}-{cparams_str}"
            )


@gin.configurable(denylist=["engine"])
class Video(base.Task):
    """Render video from the dynamic NeRF model.

    Note that for all rgb predictions, we use the quantized version for
    computing metrics such that the results are consistent when loading saved
    images afterwards.

    There are three modes for rendering videos:
        (1) Novel-view rendering, when camera_traj != 'fixed' and time_traj ==
            'fixed'.
        (2) Stabilized-view rendering, when camera_traj == 'fixed' and
            time_traj == 'replay'.
        (3) Bullet-time rendering, when camera_traj != 'fixed' and
            time_traj == 'replay'.
    """

    def __init__(
        self,
        engine: types.EngineType,
        split: Union[Sequence[str], str] = gin.REQUIRED,
        *,
        interval: Optional[int] = None,
        configs: Sequence[Dict] = gin.REQUIRED,
        use_cull_cameras: bool = True,
        correct_pose: bool = False,
    ):
        super().__init__(engine, interval=interval)
        if isinstance(split, str):
            split = [split]
        self.split = split
        self.configs = [VideoConfig(**c) for c in configs]
        self.use_cull_cameras = use_cull_cameras
        self.correct_pose = correct_pose

    @property
    def eligible(self):
        return len(self.configs) > 0

    @staticmethod
    def pad_by_fps(
        cameras: Sequence[geometry.Camera],
        metadatas: Sequence[struct.Metadata],
        dataset_fps: float,
        target_fps: float,
    ):
        T = len(metadatas)
        V = len(cameras)

        num_time_repeats = max(1, int(target_fps) // int(dataset_fps), V // T)
        num_time_skips = (
            max(1, int(dataset_fps) // int(target_fps))
            if len(metadatas) != 1
            else 1
        )
        metadatas = list(
            itertools.chain(*zip(*(metadatas,) * num_time_repeats))
        )[::num_time_skips]

        num_camera_repeats = len(metadatas) // V
        T = num_camera_repeats * V

        cameras = cameras * num_camera_repeats
        metadatas = metadatas[:T]

        return cameras, metadatas

    def start(self):
        engine = self.engine

        if not hasattr(engine, "renders_dir"):
            engine.renders_dir = osp.join(engine.work_dir, "renders")
        self.render_dir = osp.join(engine.renders_dir, "video")
        if not hasattr(engine, "eval_datasets"):
            engine.eval_datasets = dict()
        # Video dataset is private to this task.
        self.video_datasets = defaultdict(list)
        for split in self.split:
            if split not in engine.eval_datasets:
                engine.eval_datasets[split] = engine.dataset_cls.create(
                    split=split,
                    training=False,
                )
            dataset = engine.eval_datasets[split]
            for cfg in self.configs:
                traj_fn = {
                    "fixed": lambda c, **_: [c],
                    "arc": geometry.get_arc_traj,
                    "lemniscate": geometry.get_lemniscate_traj,
                }[cfg.camera_traj]

                ref_cam = dataset.parser.load_camera(
                        dataset.time_ids[cfg.time_idx],
                        dataset.camera_ids[cfg.camera_idx],
                    )
                cameras = traj_fn(
                    ref_cam,
                    lookat=dataset.lookat,
                    up=dataset.up,
                    **cfg.camera_traj_params,
                )
                
                metadatas = [
                    struct.Metadata(
                        time=np.full(tuple(cameras[0].image_shape) + (1,), t),
                        camera=np.full(
                            tuple(cameras[0].image_shape) + (1,), c
                        ),
                    )
                    for t, c in zip(
                        *{
                            "fixed": [
                                [dataset.time_ids[cfg.time_idx]],
                                [dataset.camera_ids[cfg.camera_idx]],
                            ],
                            # Replay training sequence.
                            "replay": [
                                engine.dataset.time_ids.tolist(),
                                [dataset.camera_ids[cfg.camera_idx]]
                                * dataset.num_times,
                            ],
                        }[cfg.time_traj]
                    )
                ]
                
                if str(cfg) == "Stabilized-view video":
                    cfg.fps = dataset.fps

                # metadatas = [metadatas[i] for i in range(len(metadatas)) if i >=263 and i<=282]
                cameras, metadatas = Video.pad_by_fps(
                    cameras,
                    metadatas,
                    dataset_fps=dataset.fps,
                    target_fps=cfg.fps,
                )

                if cfg.time_traj == "fixed":
                    meta_dir = osp.join(dataset.data_dir, "render_fixed_time") 
                elif cfg.time_traj == "replay":
                    meta_dir = osp.join(dataset.data_dir, "render_bullet_time") 
                for idx, (cam,meta) in enumerate(zip(cameras, metadatas)):
                    cam_dict = cam.asdict()
                    cam_dict['time'] = meta.time[0,0,0]
                    io.dump(osp.join(meta_dir, f"{str(idx).zfill(5)}.json"), cam_dict)

                self.video_datasets[split].append([cameras, metadatas])
        
        # Force cull cameras when rendering videos.
        self.prender_image = get_prender_image(
            engine.model, use_cull_cameras=self.use_cull_cameras, correct_pose=self.correct_pose,
        )

    def every_n_steps(self):
        engine = self.engine

        for split in self.split:
            dataset = self.video_datasets[split]
            for cfg, (cameras, metadatas) in zip(
                common.tqdm(
                    self.configs,
                    desc=f"* Rendering videos ({split})",
                    position=0,
                ),
                dataset,
            ):
                video = []
                for camera, metadata in zip(
                    common.tqdm(
                        cameras,
                        desc=f"* Rendering {cfg}",
                        position=1,
                        leave=False,
                    ),
                    metadatas,
                ):
                    rays = camera.pixels_to_rays(camera.get_pixels())._replace(
                        metadata=metadata
                    )
                    rendered = self.prender_image(
                        engine.pstate.optimizer.target,
                        rays,
                        key=engine.key,
                        show_pbar=False,
                    )
                    video.append(rendered["rgb"])
                io.dump(
                    osp.join(
                        self.render_dir,
                        split,
                        "checkpoints",
                        f"{cfg.short_name}.mp4",
                    ),
                    video,
                    fps=cfg.fps,
                    show_pbar=False,
                )
            logging.info(f"* Videos rendered ({split}).")

    def finalize(self):
        engine = self.engine

        for split in self.split:
            dataset = self.video_datasets[split]
            for cfg, (cameras, metadatas) in zip(
                common.tqdm(
                    self.configs,
                    desc=f"* Rendering videos ({split})",
                    position=0,
                ),
                dataset,
            ):
                video = []
                for idx, (camera, metadata) in enumerate(zip(
                    common.tqdm(
                        cameras,
                        desc=f"* Rendering {cfg}",
                        position=1,
                        leave=False,
                    ),
                    metadatas,
                )):
                    rays = camera.pixels_to_rays(camera.get_pixels())._replace(
                        metadata=metadata
                    )
                    rendered = self.prender_image(
                        engine.pstate.optimizer.target,
                        rays,
                        key=engine.key,
                        show_pbar=False,
                    )
                    video.append(rendered["rgb"])

                    io.dump(
                        osp.join(
                            self.render_dir,
                            split,
                            cfg.short_name,
                            f"{str(idx).zfill(5)}.png",
                        ),
                        image.to_quantized_float32(rendered["rgb"])
                    )
            logging.info(f"* Videos finalized ({split}).")
