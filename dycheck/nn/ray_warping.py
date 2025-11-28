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
class RayWarping(nn.Module):
    num_frames: int = 0
    eps: float = 1e-5

    def setup(self):
        self.res_rotation = self.param('res_rotation', nn.initializers.constant(self.eps) , (self.num_frames, 3))
        self.res_transl = self.param('res_transl', nn.initializers.constant(self.eps) , (self.num_frames, 3))
        self.mask_t = np.ones((self.num_frames))
        self.mask_t[0] = 0.0
        self.mask_t[-1] = 0.0
        self.mask_t = jnp.array(self.mask_t)
        self.transl_bias = 1e-2

    def __call__(self, w2cs: jnp.ndarray, time_index: jnp.ndarray) -> jnp.ndarray:
        N_rays = time_index.shape[0]
        rotation = jnp.take_along_axis(self.res_rotation, jnp.broadcast_to(time_index[:,None], (N_rays,3)), axis=0)
        transl = jnp.take_along_axis(self.res_transl, jnp.broadcast_to(time_index[:,None], (N_rays,3)), axis=0)
        mask_t = self.mask_t[time_index]
        rotation = rotation * mask_t[:,None] + self.eps 
        transl = transl * mask_t[:,None] * self.transl_bias
        R_a,t_a = barf_se3.se3_to_SE3(rotation, transl)
        R_b,t_b = w2cs[...,:3], w2cs[...,3:]
        R_new = R_b@R_a 
        t_new = (R_b@t_a+t_b)[...,0]
        return R_new, t_new