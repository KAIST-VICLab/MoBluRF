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

import os.path as osp
from collections import OrderedDict, defaultdict
from typing import Optional, Sequence, Union

import gin
import numpy as np
from absl import logging

from dycheck.utils import common, image, io, types

from .. import metrics
from . import base, utils
from .functional import get_prender_image

from easydict import EasyDict as edict
import jax.numpy as jnp
from dycheck.utils import common, struct, types
import cv2
import jax
from dycheck.geometry import barf_se3


def procrustes_analysis(X0, X1):  # [N,3] X0 is target X1 is src
    # translation
    t0 = X0.mean(axis=0, keepdims=True)
    t1 = X1.mean(axis=0, keepdims=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = jnp.sqrt((X0c**2).sum(axis=-1).mean())
    s1 = jnp.sqrt((X1c**2).sum(axis=-1).mean())
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)

    U, S, V = jnp.linalg.svd(
        (X0cs.T @ X1cs).astype(np.float64), full_matrices=False)
    R = (U @ V.T)

    if jnp.linalg.det(R) < 0:
        R = R.at[2].set(-R[2])
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3


def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@(jnp.transpose(R2, (0,2,1)))
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    
    # angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    angle = jnp.arccos(jnp.clip((trace-1)/2, -1+eps,1-eps))
    return angle

def evaluate_camera_alignment(pose_aligned_R,pose_aligned_t, pose_GT_R, pose_GT_t):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned_R, pose_aligned_t
    R_GT,t_GT = pose_GT_R, pose_GT_t
    R_error = rotation_distance(R_aligned,R_GT).mean()

    t_error = jnp.linalg.norm((t_aligned-t_GT)[...,0], axis=-1).mean()
    error = edict(R=R_error,t=t_error)
    return error

def correct_poses(w2cs, mask_t, rotation, transl):
    transl_bias = 1e-2
    eps = 1e-5

    rotation = rotation * mask_t[:,None] + eps # hack for last first pose gradient
    transl = transl * mask_t[:,None] * transl_bias

    R_a, t_a = barf_se3.se3_to_SE3(rotation, transl)
    R_b,t_b = w2cs[...,:3], w2cs[...,3:] # current pose

    R_new = R_b@R_a # this is new orientation
    t_new = (R_b@t_a+t_b)[...,0] # this is the new position

    w2cs_new = jnp.concatenate((R_new, t_new[...,None]), axis=-1)
    return w2cs_new


@gin.configurable(denylist=["engine"])
class PoseEval(base.Task):
    """Render novel view for all splits and compute metrics.

    Note that for all rgb predictions, we use the quantized version for
    computing metrics such that the results are consistent when loading saved
    images afterwards.
    """

    def __init__(
        self,
        engine: types.EngineType,
        split: Union[Sequence[str], str] = gin.REQUIRED,
        *,
        interval: Optional[int] = None,
    ):
        super().__init__(engine, interval=interval)
        if isinstance(split, str):
            split = [split]
        self.split = split

        self._step_stats = defaultdict(int)
        self.correct_pose = False

    @property
    def eligible(self):
        return self.engine.dataset.has_novel_view

    def start(self):
        engine = self.engine

        if not hasattr(engine, "renders_dir"):
            engine.renders_dir = osp.join(engine.work_dir, "renders")
        self.render_dir = osp.join(engine.renders_dir, "novel_view")
        if not hasattr(engine, "eval_datasets"):
            engine.eval_datasets = dict()
        for split in self.split:
            if split not in engine.eval_datasets:
                engine.eval_datasets[split] = engine.dataset_cls.create(
                    split=split,
                    training=False,
                )
        self.prender_image = get_prender_image(engine.model, correct_pose=self.correct_pose)

        self.compute_lpips = metrics.get_compute_lpips()

    def every_n_steps(self):
        pass

    def finalize(self):
        engine = self.engine

        for split in self.split:
            # Recreate the dataset such that the iterator is reset.
            dataset = engine.dataset_cls.create(
                split=split,
                training=False,
            )

            cameras = common.parallel_map(
                dataset.parser.load_camera,
                dataset.time_ids,
                dataset.camera_ids,
            )
            cameras_gt = common.parallel_map(
                dataset.parser.load_camera_gt,
                dataset.time_ids,
                dataset.camera_ids,
            )

            pose_graph = engine.pstate.optimizer.target['params']['pose_graph']

            # create t mask
            mask_t = np.ones((len(cameras)))
            mask_t[0] = 0.0
            mask_t[-1] = 0.0
            mask_t = jnp.array(mask_t)

            rotation = pose_graph["res_rotation"][0]
            transl = pose_graph["res_transl"][0]

            w2cs = jnp.stack([c.extrin[:3,:4] for c in cameras], axis=0)
            w2cs_gt = jnp.stack([c.extrin[:3,:4] for c in cameras_gt], axis=0)
            pred_poses = correct_poses(w2cs, mask_t, rotation, transl)

            w2cs = jnp.concatenate([w2cs[..., 0:1], -w2cs[..., 1:3], w2cs[..., 3:4]], -1)
            w2cs_gt = jnp.concatenate([w2cs_gt[..., 0:1], -w2cs_gt[..., 1:3], w2cs_gt[..., 3:4]], -1)
            pred_poses = jnp.concatenate([pred_poses[..., 0:1], -pred_poses[..., 1:3], pred_poses[..., 3:4]], -1)

            pred_poses_R = pred_poses[:,:3,:3]
            pred_poses_t = pred_poses[:,:3,3][...,None]
            pred_poses_R = pred_poses_R@jnp.transpose(pred_poses_R[0], (1,0))[None]
            pred_poses_R_c2w = jnp.transpose(pred_poses_R, (0,2,1))
            pred_o = -pred_poses_R_c2w@(pred_poses_t)
            pred_o = pred_o - pred_o[0][None]


            w2cs_R = w2cs[:,:3,:3]
            w2cs_t = w2cs[:,:3,3][...,None]
            w2cs_R = w2cs_R@jnp.transpose(w2cs_R[0], (1,0))[None]
            w2cs_R_c2w = jnp.transpose(w2cs_R, (0,2,1))
            w2cs_t_o = -w2cs_R_c2w@(w2cs_t)
            w2cs_t_o = w2cs_t_o - w2cs_t_o[0][None]

            w2cs_gt_R = w2cs_gt[:,:3,:3]
            w2cs_gt_t = w2cs_gt[:,:3,3][...,None]
            w2cs_gt_R = w2cs_gt_R@jnp.transpose(w2cs_gt_R[0], (1,0))[None]
            w2cs_gt_R_c2w = jnp.transpose(w2cs_gt_R, (0,2,1))
            w2cs_gt_t_o = -w2cs_gt_R_c2w@(w2cs_gt_t)
            w2cs_gt_t_o = w2cs_gt_t_o - w2cs_gt_t_o[0][None]

            cam_error = evaluate_camera_alignment(pred_poses_R, pred_o, w2cs_gt_R, w2cs_gt_t_o)
            # cam_error_no_optim = evaluate_camera_alignment(w2cs_R,w2cs_t_o,w2cs_gt_R, w2cs_gt_t_o)


            mean_metrics_dict = {
                'r_our':float(cam_error.R),
                't_our':float(cam_error.t)/dataset.scale,
                # 'r_max':float(cam_error_no_optim.R),
                # 't_max':float(cam_error_no_optim.t),
            }

            logging.info(
                (
                    f"* Mean novel view metrics ({split}):\n"
                    f"{utils.format_dict(mean_metrics_dict)}"
                )
            )