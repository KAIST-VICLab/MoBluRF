from typing import Literal, Optional, Tuple

import gin
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

from dycheck.nn import functional as F
from dycheck.utils import common, types
from dycheck.geometry import barf_se3

@gin.configurable(denylist=["name"])
class PoseGraph_Blurry(nn.Module):
    num_frames: int = 0
    eps: float = jnp.finfo(jnp.float32).eps
    blurry_step: int = 6

    def setup(self):
        self.blurry_res_rotation = self.param('blurry_res_rotation', nn.initializers.constant(self.eps) , (self.num_frames, 3* self.blurry_step))
        self.blurry_res_transl = self.param('blurry_res_transl', nn.initializers.constant(self.eps) , (self.num_frames, 3* self.blurry_step))

        # create t mask
        self.mask_t = np.ones((self.num_frames))
        self.mask_t[0] = 0.0
        self.mask_t[-1] = 0.0
        self.mask_t = jnp.array(self.mask_t)

        # translation bias, trans < rotation refine !
        self.transl_bias = 1e-2

    def __call__(self, w2cs: jnp.ndarray, time_index: jnp.ndarray) -> jnp.ndarray:
        N_rays = time_index.shape[0]
        rotation = jnp.take_along_axis(self.blurry_res_rotation, jnp.broadcast_to(time_index[:,None], (N_rays, 3 * self.blurry_step)), axis=0)
        transl = jnp.take_along_axis(self.blurry_res_transl, jnp.broadcast_to(time_index[:,None], (N_rays, 3 * self.blurry_step)), axis=0)
        mask_t = self.mask_t[time_index]

        # mask out the first and last refine pose
        rotation = rotation * mask_t[:,None] + self.eps # hack for last first pose gradient
        transl = transl * mask_t[:,None] * self.transl_bias

        rotation = jnp.reshape(rotation, (-1, 3))
        transl = jnp.reshape(transl, (-1, 3))

        R_a, t_a = barf_se3.se3_to_SE3(rotation, transl)
        w2cs_expand = jnp.reshape(jnp.broadcast_to(w2cs[:,None], (w2cs.shape[0], self.blurry_step) + (w2cs.shape[1:])), (-1, 3, 4))
        R_b,t_b = w2cs_expand[...,:3], w2cs_expand[...,3:] # current pose

        R_new = R_b@R_a # this is new orientation (B*blurry, 3, 3)
        t_new = (R_b@t_a+t_b)[...,0] # this is the new position (B*blurry, 3, 1)
        return R_new, t_new