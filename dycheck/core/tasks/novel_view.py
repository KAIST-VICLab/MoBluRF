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
import jax 

from dycheck.utils import common, image, io, types

from .. import metrics
from . import base, utils
from .functional import get_prender_image

from easydict import EasyDict as edict
import jax.numpy as jnp
from dycheck.utils import common, struct, types
import cv2

def get_tOF(pre_gt_grey, gt_grey, pre_output_grey, output_grey):
    target_OF = cv2.calcOpticalFlowFarneback(pre_gt_grey, gt_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    output_OF = cv2.calcOpticalFlowFarneback(pre_output_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    target_OF, ofy, ofx = crop_8x8(target_OF)
    output_OF, ofy, ofx = crop_8x8(output_OF)

    OF_diff = np.absolute(target_OF - output_OF)
    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # l1 vector norm

    return OF_diff.mean()


def crop_8x8(img):
    ori_h = img.shape[0]
    ori_w = img.shape[1]

    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32

    while (h > ori_h - 16):
        h = h - 32
    while (w > ori_w - 16):
        w = w - 32

    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    crop_img = img[y:y + h, x:x + w]
    return crop_img, y, x


@gin.configurable(denylist=["engine"])
class NovelView(base.Task):
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
        force: bool = False,
    ):
        super().__init__(engine, interval=interval)
        if isinstance(split, str):
            split = [split]
        self.split = split

        self._step_stats = defaultdict(int)
        self.correct_pose = False
        self.force = force

    @property
    def eligible(self):
        return self.engine.dataset.has_novel_view or self.force

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
        engine = self.engine
        for split in self.split:
            # Recreate the dataset such that the iterator is reset.
            dataset = engine.dataset_cls.create(
                split=split,
                training=False,
            )
            metrics_dicts = []
            pbar = common.tqdm(
                range(len(dataset)),
                desc=f"* Rendering novel views ({split})",
            )
            for i, batch in zip(pbar, dataset):
                if i%16 != 0:
                    continue
                frame_name = dataset.frame_names[i]
                rendered = self.prender_image(
                    engine.pstate.optimizer.target,
                    batch["rays"],
                    key=engine.key,
                    show_pbar=False,
                )
                rgb = image.to_quantized_float32(batch["rgb"])
                mask = image.to_quantized_float32(batch["mask"])
                pred_rgb = image.to_quantized_float32(rendered["rgb"])
                pred_mask =  image.to_quantized_float32(rendered["raw_mask"])
                pred_dynamic_rgb = image.to_quantized_float32(rendered["dynamic_rgb"])
                pred_static_rgb = image.to_quantized_float32(rendered["static_rgb"])
                pred_depth = image.to_quantized_float32(rendered["depth"])

                metrics_dict = OrderedDict(
                    {
                        "frame_name": frame_name,
                        "psnr": metrics.compute_psnr(
                            rgb, pred_rgb, mask
                        ).item(),
                        "ssim": metrics.compute_ssim(
                            rgb, pred_rgb, mask
                        ).item(),
                        "lpips": self.compute_lpips(
                            rgb, pred_rgb, mask
                        ).item(),
                    }
                )
                combined_imgs = [rgb, pred_rgb]
                if "covisible" in batch:
                    covisible = image.to_quantized_float32(batch["covisible"])
                    metrics_dict.update(
                        **{
                            "mpsnr": metrics.compute_psnr(
                                rgb, pred_rgb, covisible
                            ).item(),
                            "mssim": metrics.compute_ssim(
                                rgb, pred_rgb, covisible
                            ).item(),
                            "mlpips": self.compute_lpips(
                                rgb, pred_rgb, covisible
                            ).item(),
                        }
                    )
                    # Mask out the non-covisible region by white color.
                    covisible_pred_rgb = (
                        covisible * pred_rgb
                        + (1 - covisible) * (1 + pred_rgb) / 2
                    )
                    combined_imgs.append(covisible_pred_rgb)
                pbar.set_description(
                    f"* Rendering novel view ({split}), "
                    + ", ".join(
                        f"{k}: {v:.3f}"
                        for k, v in metrics_dict.items()
                        if k != "frame_name"
                    )
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + ".png"),
                    np.concatenate(combined_imgs, axis=1),
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_mask.png"),
                    pred_mask,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_pred.png"),
                    pred_rgb,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_pred_static.png"),
                    pred_static_rgb,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_pred_dy.png"),
                    pred_dynamic_rgb,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_depth.png"),
                    pred_depth,
                )
                # Skip logging to tensorboard bc it's a lot of images.
                metrics_dicts.append(metrics_dict)
            metrics_dict = common.tree_collate(metrics_dicts)
            io.dump(
                osp.join(self.render_dir, split, "metrics_dict.npz"),
                **metrics_dict,
            )
            mean_metrics_dict = {
                k: float(v.mean())
                for k, v in metrics_dict.items()
                if k != "frame_name"
            }
            io.dump(
                osp.join(self.render_dir, split, "mean_metrics_dict.json"),
                mean_metrics_dict,
                sort_keys=False,
            )
            logging.info(
                (
                    f"* Mean novel view metrics ({split}):\n"
                    f"{utils.format_dict(mean_metrics_dict)}"
                )
            )

    def finalize(self):
        engine = self.engine
        for split in self.split:
            # Recreate the dataset such that the iterator is reset.
            dataset = engine.dataset_cls.create(
                split=split,
                training=False,
            )
            metrics_dicts = []
            pbar = common.tqdm(
                range(len(dataset)),
                desc=f"* Rendering novel views ({split})",
            )

            tofs = []

            pre_gt_grey, pre_output_grey = None, None
            for i, batch in zip(pbar, dataset):
                frame_name = dataset.frame_names[i]
                rendered = self.prender_image(
                    engine.pstate.optimizer.target,
                    batch["rays"],
                    key=engine.key,
                    show_pbar=False,
                )
                rgb = image.to_quantized_float32(batch["rgb"])
                mask = image.to_quantized_float32(batch["mask"])

                pred_rgb = image.to_quantized_float32(rendered["rgb"])
                pred_mask =  image.to_quantized_float32(rendered["raw_mask"])
                pred_dynamic_rgb = image.to_quantized_float32(rendered["dynamic_rgb"])
                pred_static_rgb = image.to_quantized_float32(rendered["static_rgb"])
                pred_depth = image.to_quantized_float32(rendered["depth"])

                gt_grey = cv2.cvtColor((rgb*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                output_grey = cv2.cvtColor((pred_rgb*255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                if pre_gt_grey is not None:
                    tOF = get_tOF(pre_gt_grey, gt_grey, pre_output_grey, output_grey)
                else:
                    tOF = -1.0
                tofs.append(tOF)
                metrics_dict = OrderedDict(
                    {
                        "frame_name": frame_name,
                        "psnr": metrics.compute_psnr(
                            rgb, pred_rgb, mask
                        ).item(),
                        "ssim": metrics.compute_ssim(
                            rgb, pred_rgb, mask
                        ).item(),
                        "lpips": self.compute_lpips(
                            rgb, pred_rgb, mask
                        ).item(),
                        
                    }
                )
                combined_imgs = [rgb, pred_rgb]
                if "covisible" in batch:
                    covisible = image.to_quantized_float32(batch["covisible"])
                    metrics_dict.update(
                        **{
                            "mpsnr": metrics.compute_psnr(
                                rgb, pred_rgb, covisible
                            ).item(),
                            "mssim": metrics.compute_ssim(
                                rgb, pred_rgb, covisible
                            ).item(),
                            "mlpips": self.compute_lpips(
                                rgb, pred_rgb, covisible
                            ).item(),
                        }
                    )
                    # Mask out the non-covisible region by white color.
                    covisible_pred_rgb = (
                        covisible * pred_rgb
                        + (1 - covisible) * (1 + pred_rgb) / 2
                    )
                    combined_imgs.append(covisible_pred_rgb)
                pbar.set_description(
                    f"* Rendering novel view ({split}), "
                    + ", ".join(
                        f"{k}: {v:.3f}"
                        for k, v in metrics_dict.items()
                        if k != "frame_name"
                    )
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + ".png"),
                    np.concatenate(combined_imgs, axis=1),
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_mask.png"),
                    pred_mask,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_pred.png"),
                    pred_rgb,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_pred_static.png"),
                    pred_static_rgb,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_pred_dy.png"),
                    pred_dynamic_rgb,
                )
                io.dump(
                    osp.join(self.render_dir, split, frame_name + "_depth.png"),
                    pred_depth,
                )
                # Skip logging to tensorboard bc it's a lot of images.
                metrics_dicts.append(metrics_dict)
                
                if i < len(dataset)-1:
                    if int(dataset.frame_names[i+1].split('_')[-1]) == (int(dataset.frame_names[i].split('_')[-1]) + 1):
                        pre_gt_grey = gt_grey
                        pre_output_grey = output_grey
                    else:
                        pre_gt_grey = None
                        pre_output_grey = None
            metrics_dict = common.tree_collate(metrics_dicts)
            io.dump(
                osp.join(self.render_dir, split, "metrics_dict.npz"),
                **metrics_dict,
            )
            mean_metrics_dict = {
                k: float(v.mean())
                for k, v in metrics_dict.items()
                if k != "frame_name"
            }
            tofs = np.array(tofs)
            mean_metrics_dict['tof'] = float(tofs[tofs >= 0.0].mean())

            io.dump(
                osp.join(self.render_dir, split, "mean_metrics_dict.json"),
                mean_metrics_dict,
                sort_keys=False,
            )
            logging.info(
                (
                    f"* Mean novel view metrics ({split}):\n"
                    f"{utils.format_dict(mean_metrics_dict)}"
                )
            )
