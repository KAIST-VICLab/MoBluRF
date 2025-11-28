#!/usr/bin/env python3
#
# File   : structures.py
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

from typing import NamedTuple, Optional

import flax
import gin
import jax.numpy as jnp
import optax

from . import types


class Metadata(NamedTuple):
    time: Optional[jnp.ndarray] = None
    camera: Optional[jnp.ndarray] = None
    time_to: Optional[jnp.ndarray] = None


class Rays(NamedTuple):
    origins: jnp.ndarray
    directions: jnp.ndarray
    pixels: jnp.ndarray
    local_directions: Optional[jnp.ndarray] = None
    radii: Optional[jnp.ndarray] = None
    metadata: Optional[Metadata] = None

    near: Optional[jnp.ndarray] = None
    far: Optional[jnp.ndarray] = None


class Samples(NamedTuple):
    xs: jnp.ndarray
    directions: jnp.ndarray
    cov_diags: Optional[jnp.ndarray] = None
    metadata: Optional[Metadata] = None

    tvals: Optional[jnp.ndarray] = None


class ExtraParams(NamedTuple):
    warp_alpha: Optional[jnp.ndarray] = None
    ambient_alpha: Optional[jnp.ndarray] = None
    w_time: Optional[jnp.ndarray] = None
    current_step: Optional[jnp.ndarray] = None


@flax.struct.dataclass
class TrainState(object):
    optimizer: flax.optim.Optimizer
    # pose_optimizer: Optional[flax.optim.Optimizer] = None


@flax.struct.dataclass
class TrainScalars(object):
    lr: float
    lr_pose: float
    bkgd: float = 0
    depth: float = 0
    dist: float = 0
    entropy: float = 0
    w_fine: int = 0
    w_smooth: float = 0
    w_normal: float = 0
    current_step: int = 0


@gin.configurable()
@flax.struct.dataclass
class TrainSchedules(object):
    # Training parameters for the losses.
    lr_sched: types.ScheduleType = gin.REQUIRED
    lr_pose_sched: Optional[types.ScheduleType] = None
    bkgd_sched: Optional[types.ScheduleType] = None
    depth_sched: Optional[types.ScheduleType] = None
    dist_sched: Optional[types.ScheduleType] = None
    # Extra parameters for the model.
    warp_alpha_sched: Optional[types.ScheduleType] = None
    ambient_alpha_sched: Optional[types.ScheduleType] = None
    entropy_sched: Optional[types.ScheduleType] = None
    fine_sched: Optional[types.ScheduleType] = None
    time_sched: Optional[types.ScheduleType] = None
    smooth_sched: Optional[types.ScheduleType] = None
    normal_sched: Optional[types.ScheduleType] = None

    def eval_scalars(self, step: int) -> TrainScalars:
        return TrainScalars(
            lr=self.lr_sched(step),
            lr_pose=self.lr_pose_sched(step) if self.lr_pose_sched else 0,
            bkgd=self.bkgd_sched(step) if self.bkgd_sched else 0,
            depth=self.depth_sched(step) if self.depth_sched else 0,
            dist=self.dist_sched(step) if self.dist_sched else 0,
            entropy=self.entropy_sched(step) if self.entropy_sched else 0,
            w_fine=self.fine_sched(step) if self.fine_sched else 0,
            w_smooth=self.smooth_sched(step) if self.smooth_sched else 0,
            w_normal=self.normal_sched(step) if self.normal_sched else 0,
            current_step=step,
        )

    def eval_extra_params(self, step: int) -> ExtraParams:
        return ExtraParams(
            warp_alpha=self.warp_alpha_sched(step)
            if self.warp_alpha_sched
            else 0,
            ambient_alpha=self.ambient_alpha_sched(step)
            if self.ambient_alpha_sched
            else 0,
            w_time=self.time_sched(step) if self.time_sched else 0,
            current_step=step,
        )
