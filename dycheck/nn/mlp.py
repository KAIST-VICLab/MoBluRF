#!/usr/bin/env python3
#
# File   : mlp.py
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

from typing import Literal, Optional, Tuple, Any, Callable, Iterable, List, Optional, Sequence, Union

import gin
import jax
import jax.numpy as jnp
from jax import lax

from flax import linen as nn
from flax.linen.initializers import lecun_normal
from flax.linen.initializers import zeros

from dycheck.nn import functional as F
from dycheck.utils import common, types

import numpy as np
# import time
from dycheck.geometry import barf_se3

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]

default_kernel_init = lecun_normal()


def grid_sample_interp(input, grid, mode='bilinear', padding_mode='zeros'):
    """
    Perform 1D grid sample on input tensor.
    
    Args:
        input (jax.numpy.ndarray): Input tensor of shape (width, channels).
        grid (jax.numpy.ndarray): 1D grid tensor of shape (batch, samples, 1).
        mode (str): Interpolation mode ('bilinear' or 'nearest').
        padding_mode (str): Padding mode for out-of-bound indices ('zeros' or 'border').
    
    Returns:
        jax.numpy.ndarray: Sampled tensor.
    """
    in_width, rank = input.shape
    #out_height, out_width = grid.shape[1], grid.shape[2]

    # Reshape grid to (batch, out_height * out_width, 2)
    #grid = jnp.reshape(grid, (batch_size, -1, 2))

    # Normalize grid values to range [-1, 1]
    #grid = 2 * grid / jnp.array([in_width - 1, in_height - 1]) - 1

    grid = grid[:, 0, 0]

    # Compute indices for interpolation
    if padding_mode == 'border':
        coords = lax.clamp(grid, -1.0, 1.0)
    elif padding_mode == 'zeros':
        coords = jnp.clip(grid, -1.0, 1.0)

    coords = 0.5 * (coords + 1) * (in_width - 1)

    # Compute indices for bilinear interpolation
    x0 = jnp.floor(coords).astype(jnp.int32)
    x1 = x0 + 1

    # Clip indices to ensure they are within bounds
    x0 = lax.clamp(0, x0, in_width - 1)
    x1 = lax.clamp(0, x1, in_width - 1)
    
    # Gather pixel values at the computed indices
    Ia = input[x0, :]
    Ib = input[x1, :]

    # Compute weights for bilinear interpolation
    wa = jnp.tile((x1 - coords)[:, jnp.newaxis], (1, rank))
    wb = jnp.tile((coords - x0)[:, jnp.newaxis], (1, rank))

    # Perform bilinear interpolation
    if mode == 'bilinear':
        output = wa * Ia + wb * Ib

    return output # [N, R]


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
    """
    Perform 1D grid sample on input tensor.
    
    Args:
        input (jax.numpy.ndarray): Input tensor of shape (width, channels).
        grid (jax.numpy.ndarray): 1D grid tensor of shape (batch, samples, 1).
        mode (str): Interpolation mode ('bilinear' or 'nearest').
        padding_mode (str): Padding mode for out-of-bound indices ('zeros' or 'border').
    
    Returns:
        jax.numpy.ndarray: Sampled tensor.
    """
    in_width, rank = input.shape
    #out_height, out_width = grid.shape[1], grid.shape[2]

    # Reshape grid to (batch, out_height * out_width, 2)
    #grid = jnp.reshape(grid, (batch_size, -1, 2))

    # Normalize grid values to range [-1, 1]
    #grid = 2 * grid / jnp.array([in_width - 1, in_height - 1]) - 1

    grid = grid[:, 0, :]

    # Compute indices for interpolation
    if padding_mode == 'border':
        coords = lax.clamp(grid, -1.0, 1.0)
    elif padding_mode == 'zeros':
        coords = jnp.clip(grid, -1.0, 1.0)

    coords = 0.5 * (coords + 1) * (in_width - 1)

    # Compute indices for bilinear interpolation
    x0 = jnp.floor(coords).astype(jnp.int32)
    x1 = x0 + 1

    # Clip indices to ensure they are within bounds
    x0 = lax.clamp(0, x0, in_width - 1)
    x1 = lax.clamp(0, x1, in_width - 1)
    
    # Gather pixel values at the computed indices
    Ia = input[x0, :]
    Ib = input[x1, :]

    # Compute weights for bilinear interpolation
    wa = jnp.tile((x1 - coords)[:, :, jnp.newaxis], (1, 1, rank))
    wb = jnp.tile((coords - x0)[:, :, jnp.newaxis], (1, 1, rank))

    # Perform bilinear interpolation
    if mode == 'bilinear':
        output = wa * Ia + wb * Ib

    return output # [N, 7, R]


@gin.configurable(denylist=["name"])
class MLP(nn.Module):
    # FIXME: This MLP has different skip scheme compared to all the other
    # NeRFs. Repro however requires it.

    depth: int = gin.REQUIRED
    width: int = gin.REQUIRED

    hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
    hidden_activation: types.Activation = nn.relu

    output_init: types.Initializer = jax.nn.initializers.glorot_uniform()
    output_channels: int = 0
    output_activation: types.Activation = F.activations.identity

    use_bias: bool = True
    skips: Tuple[int] = tuple()

    def setup(self):
        # NOTE(Hang Gao @ 03/06): Do static setup rather than compact because
        # we might use this function for root-finding, which requires this part
        # to be pre-declared.
        layers = [
            nn.Dense(
                self.width,
                use_bias=self.use_bias,
                kernel_init=self.hidden_init,
                name=f"hidden_{i}",
            )
            for i in range(self.depth)
        ]
        self.layers = layers

        if self.output_channels > 0:
            self.logit_layer = nn.Dense(
                self.output_channels,
                use_bias=self.use_bias,
                kernel_init=self.output_init,
                name="logit",
            )

    def __call__(self, xs: jnp.ndarray) -> jnp.ndarray:
        inputs = xs
        for i in range(self.depth):
            layer = self.layers[i]
            if i in self.skips:
                xs = jnp.concatenate([xs, inputs], axis=-1)
            #start_time = time.time()
            xs = layer(xs)
            #print("time for single layer (mlp) : ", time.time()-start_time)
            xs = self.hidden_activation(xs)

        if self.output_channels > 0:
            xs = self.logit_layer(xs)
            xs = self.output_activation(xs)

        return xs


@gin.configurable(denylist=["name"])
class NeRFMLP(nn.Module):
    trunk_depth: int = 8
    trunk_width: int = 256

    sigma_depth: int = 0
    sigma_width: int = 128
    sigma_channels: int = 1

    blending_depth: int = 0
    blending_width: int = 128
    blending_channels: int = 1

    rgb_depth: int = 1
    rgb_width: int = 128
    rgb_channels: int = 3

    hidden_activation: types.Activation = nn.relu
    skips: Tuple[int] = (4,)
    pred_blending: bool = False

    @nn.compact
    def __call__(
        self,
        xs: jnp.ndarray,
        trunk_conditions: Optional[jnp.ndarray] = None,
        rgb_conditions: Optional[jnp.ndarray] = None,
        return_fields: Tuple[Literal["point_sigma", "point_rgb", "point_blending"]] = (
            "point_sigma",
            "point_rgb",
            "point_blending"
        ),
    ) -> jnp.ndarray:
        trunk_mlp = MLP(
            depth=self.trunk_depth,
            width=self.trunk_width,
            hidden_activation=self.hidden_activation,
            skips=self.skips,
            name="trunk",
        )
        sigma_mlp = MLP(
            depth=self.sigma_depth,
            width=self.sigma_width,
            hidden_activation=self.hidden_activation,
            output_channels=self.sigma_channels,
            name="sigma",
        )

        if self.pred_blending:
            blending_mlp = MLP(
                depth=self.blending_depth,
                width=self.blending_width,
                hidden_activation=self.hidden_activation,
                output_channels=self.blending_channels,
                name="blending",
            )

        rgb_mlp = MLP(
            depth=self.rgb_depth,
            width=self.rgb_width,
            hidden_activation=self.hidden_activation,
            output_channels=self.rgb_channels,
            name="rgb",
        )

        trunk = xs
        if trunk_conditions is not None:
            trunk = jnp.concatenate([trunk, trunk_conditions], axis=-1)
        trunk = trunk_mlp(trunk)

        sigma = sigma_mlp(trunk)
        #  print(sigma.mean(), trunk.mean(), xs.mean())

        if self.pred_blending:
            blending_w = blending_mlp(trunk)

        if rgb_conditions is not None:
            # Use one extra layer to align with original NeRF model.
            trunk = nn.Dense(
                trunk_mlp.width,
                kernel_init=trunk_mlp.hidden_init,
                name=f"bottleneck",
            )(trunk)
            trunk = jnp.concatenate([trunk, rgb_conditions], axis=-1)
        rgb = rgb_mlp(trunk)

        out = {"point_sigma": sigma, "point_rgb": rgb}

        if self.pred_blending:
            out["point_blending"] = blending_w

        out = common.traverse_filter(
            out, return_fields=return_fields, inplace=True
        )
        return out

@gin.configurable(denylist=["name"])
class LocalObjectMotionMLP(nn.Module):
    trunk_depth: int = 6
    trunk_width: int = 128

    raytime_depth: int = 0
    raytime_width: int = 128
    raytime_channels: int = 6

    hidden_activation: types.Activation =  F.activations.identity
    skips: Tuple[int] = ()
    output_bias: float = 1e-2
    transl_bias: float = 1e-2
    eps: float = jnp.finfo(jnp.float32).eps
    num_frames: int = 0

    def setup(self):
        # create t mask
        self.mask_t = np.ones((self.num_frames))
        self.mask_t[0] = 0.0
        self.mask_t[-1] = 0.0
        self.mask_t = jnp.array(self.mask_t)

    @nn.compact
    def __call__(
        self,
        xs: jnp.ndarray,
        time_index: jnp.ndarray,
        w2cs: jnp.ndarray,
        motion_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        
        trunk_mlp = MLP(
            depth=self.trunk_depth,
            width=self.trunk_width,
            hidden_activation=self.hidden_activation,
            skips=self.skips,
            name="trunk",
        )
        raytime_mlp = MLP(
            depth=self.raytime_depth,
            width=self.raytime_width,
            hidden_activation=self.hidden_activation,
            output_channels=self.raytime_channels,
            name="raytime",
        )

        trunk = xs
        trunk = trunk_mlp(trunk)

        motion_blur_rt = raytime_mlp(trunk) * self.output_bias * motion_mask # B,6
        rotation = motion_blur_rt[...,:3]
        transl = motion_blur_rt[...,3:]

        mask_t = self.mask_t[time_index]

        rotation = rotation * mask_t[:,None] + self.eps # hack for last first pose gradient
        transl = transl * mask_t[:,None] * self.transl_bias + self.eps
        
        R_a, t_a = barf_se3.se3_to_SE3(rotation, transl)
        R_b,t_b = w2cs[...,:3], w2cs[...,3:] # current pose

        R_new = R_b@R_a # this is new orientation (B*blurry, 3, 3)
        t_new = (R_b@t_a+t_b)[...,0] # this is the new position (B*blurry, 3, 1)
        return R_new, t_new