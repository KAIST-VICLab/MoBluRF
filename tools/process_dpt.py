import dataclasses
import os.path as osp
from collections import defaultdict
from typing import Callable, Sequence

import gin
import numpy as np
from absl import app, flags, logging

from dycheck import core, processors
from dycheck.core.tasks import utils
from dycheck.datasets import Parser
from dycheck.utils import common, io, image

flags.DEFINE_multi_string(
    "gin_configs", None, "Gin config files.", required=True
)
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
FLAGS = flags.FLAGS


@gin.configurable(module="process_dpt")
@dataclasses.dataclass
class Config(object):
    parser_cls: Callable[..., Parser] = gin.REQUIRED
    splits: Sequence[str] = gin.REQUIRED


def main(_):
    logging.info(f"*** Loading Gin configs from: {FLAGS.gin_configs}.")
    core.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs,
        bindings=FLAGS.gin_bindings,
        skip_unknown=True,
        master=False,
    )

    config_str = gin.config_str()
    logging.info(f"*** Configuration:\n{config_str}")

    config = Config()

    logging.info("*** Starting processing dpt.")
    parser = config.parser_cls()

    dpt_path = osp.join(parser.data_dir, "blurry_depth/2x")
    viz_dpt_path = osp.join(parser.data_dir, "blurry_depth_img/2x")

    normal_path = osp.join(parser.data_dir, "blurry_normal/2x")
    viz_normal_path = osp.join(parser.data_dir, "blurry_normal_img/2x")

    for split in config.splits:
        _, time_ids, camera_ids = parser.load_split(split)
        rgbs = np.array(
            common.parallel_map(parser.load_rgba, time_ids, camera_ids)
        )[..., :3]

        cameras = common.parallel_map(parser.load_camera, time_ids, camera_ids)

        compute_dpt_disp = processors.get_compute_dpt_disp()
        dpt_disps = np.array(
            [
                compute_dpt_disp(rgb)
                for rgb in common.tqdm(rgbs, desc="* Compute DPT depth")
            ]
        ).astype(np.float32)

    max_disps = dpt_disps.reshape(dpt_disps.shape[0], -1).max(axis=-1)[:,None,None,None]
    min_disps = dpt_disps.reshape(dpt_disps.shape[0], -1).min(axis=-1)[:,None,None,None]

    norm_disps = (dpt_disps - min_disps) / (max_disps - min_disps + 1e-8)
        
    for i, (camera_id, time_id) in enumerate(zip(camera_ids, time_ids)):
        save_path = osp.join(dpt_path, '{}_{}.npy'.format(camera_id, str(time_id).zfill(5)))
        save_img_path = osp.join(viz_dpt_path, '{}_{}.png'.format(camera_id, str(time_id).zfill(5)))

        save_normal_path = osp.join(normal_path, '{}_{}.npy'.format(camera_id, str(time_id).zfill(5)))
        save_normal_img_path = osp.join(viz_normal_path, '{}_{}.png'.format(camera_id, str(time_id).zfill(5)))

        # compute normals from disp
        camera = cameras[i]
        pixels = camera.get_pixels()
        baseline = 1.0
        disp = dpt_disps[i]
        z = camera.scale_factor_x * baseline / (disp + 0.0001)


        y = (pixels[..., 1] - camera.principal_point_y) / camera.scale_factor_y
        x = (
            pixels[..., 0] - camera.principal_point_x - y * camera.skew
        ) / camera.scale_factor_x
        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        coords = viewdirs * z
        coords = (coords.transpose(2,0,1))[None]

        dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
        dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
        dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
        dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
        dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
        dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

        dxdu = np.concatenate([dxdu, dxdu[...,:, -1:]], axis=-1)
        dydu = np.concatenate([dydu, dydu[...,:, -1:]], axis=-1)
        dzdu = np.concatenate([dzdu, dzdu[...,:, -1:]], axis=-1)

        dxdv = np.concatenate([dxdv, dxdv[..., -1:, :]], axis=-2)
        dydv = np.concatenate([dydv, dydv[..., -1:, :]], axis=-2)
        dzdv = np.concatenate([dzdv, dzdv[..., -1:, :]], axis=-2)
        
        n_x = dydv * dzdu - dydu * dzdv
        n_y = dzdv * dxdu - dzdu * dxdv
        n_z = dxdv * dydu - dxdu * dydv
        
        n = np.stack([n_x, n_y, n_z], axis=-3)
        normal_vec = (n / np.linalg.norm(n, axis=-3)[:,None]).squeeze(0).transpose(1,2,0)

        io.dump(save_path, dpt_disps[i])
        io.dump(save_img_path, image.to_quantized_float32(norm_disps[i]))

        io.dump(save_normal_path, normal_vec)
        io.dump(save_normal_img_path, image.to_quantized_float32(normal_vec))



if __name__ == "__main__":
    app.run(main)