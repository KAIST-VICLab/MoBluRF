#!/usr/bin/env python3
#
# File   : rendering.py
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

from typing import Dict, Optional

import jax.numpy as jnp
from jax import random
import jax

from dycheck.utils import struct, types
from einshape import jax_einshape as einshape


def perturb_logits(
    key: types.PRNGKey,
    logits: Dict[str, jnp.ndarray],
    use_randomized: bool,
    noise_std: Optional[float],
) -> Dict[str, jnp.ndarray]:
    """Regularize the sigma prediction by adding gaussian noise.

    Args:
        key (types.PRNGKey): A random number generator.
        logits (Dict[str, Any]): A dictionary holding at least "sigma".
        use_randomized (Dict[str, jnp.ndarray]): Add noise only if
            use_randomized is True and noise_std is bigger than 0,
        noise_std (Optional[float]): Standard dev of noise added to regularize
            sigma output.

    Returns:
        logits: Updated
            logits.
    """
    if use_randomized and noise_std is not None and noise_std > 0.0:
        assert "point_sigma" in logits
        key = random.split(key)[1]
        noise = (
            random.normal(key, logits["point_sigma"].shape, dtype=logits.dtype)
            * noise_std
        )
        logits["point_sigma"] += noise
    return logits


def volrend(
    out: Dict[str, jnp.ndarray],
    samples: struct.Samples,
    bkgd_rgb: jnp.ndarray,
    use_sample_at_infinity: bool,
    eps: float = 1e-10,
) -> Dict[str, jnp.ndarray]:
    """Render through volume by numerical integration.

    Args:
        out (Dict[str, jnp.ndarray]): A dictionary holding at least
            "point_sigma" (..., S, 1) and "point_rgb" (..., S, 3).
        samples (struct.Samples): Samples to render of shape (..., S).
        bkgd_rgb (jnp.ndarray): Background color of shape (3,).
        use_sample_at_infinity (bool): Whether to sample at infinity.

    Returns:
        Dict[str, jnp.ndarray]: rendering results.
    """
    assert samples.tvals is not None
    assert "point_sigma" in out and "point_rgb" in out
    batch_shape = samples.xs.shape[:-1]

    # TODO(keunhong): Remove this hack.
    # NOTE(Hang Gao @ 07/15): Actually needed by Nerfies & HyperNeRF to always
    # stop by the scene bound otherwise it will not when trained without depth.
    last_sample_t = 1e10 if use_sample_at_infinity else 1e-19

    # (..., S, 1)

    dists = jnp.concatenate(
        [
            samples.tvals[..., 1:, :] - samples.tvals[..., :-1, :],
            jnp.broadcast_to(jnp.array([last_sample_t]), batch_shape[:-1] + (1, 1)),
        ],
        -2,
    )
    dists = dists * jnp.linalg.norm(samples.directions, axis=-1, keepdims=True)

    # (..., S, 1)
    alpha = 1 - jnp.exp(-out["point_sigma"] * dists)
    # Prepend a 1 to make this an 'exclusive' cumprod as in `tf.math.cumprod`.
    # (..., S, 1)
    trans = jnp.concatenate(
        [
            jnp.ones_like(alpha[..., :1, :], alpha.dtype),
            jnp.cumprod(1 - alpha[..., :-1, :] + eps, axis=-2),
        ],
        axis=-2,
    )
    # (..., S, 1)
    weights = alpha * trans

    # (..., 1)
    acc = (
        weights[..., :-1, :].sum(axis=-2)
        if use_sample_at_infinity
        else weights.sum(axis=-2)
    )

    # (..., 1)
    # Avoid 0/0 case.
    depth = (
        weights[..., :-1, :] * samples.tvals[..., :-1, :]
        if use_sample_at_infinity
        else weights * samples.tvals
    ).sum(axis=-2) / acc.clip(1e-12)
    # This nan_to_num trick from Jon does not really work for the 0/0 case and
    # will cause NaN gradient.
    depth = jnp.clip(
        jnp.nan_to_num(depth, nan=jnp.inf),
        samples.tvals[..., 0, :],
        samples.tvals[..., -1, :],
    )

    # (..., 3)
    rgb = (weights * out["point_rgb"]).sum(axis=-2)
    rgb = rgb + bkgd_rgb * (1 - acc)

    # ref rendering
    if 'point_ref_rgb' in out:
        ref_rgb = (jax.lax.stop_gradient(weights) * out["point_ref_rgb"]).sum(axis=-2)
        ref_rgb = ref_rgb + bkgd_rgb * (1 - jax.lax.stop_gradient(acc))
        ref_rgb = einshape("nbc->bnc", ref_rgb)

        ref_mask = (jax.lax.stop_gradient(weights) * out["point_ref_mask"]).sum(axis=-2)
        ref_mask = einshape("nbc->bnc", ref_mask)
        out = {
            "ref_rgb": ref_rgb,
            "ref_mask": jax.lax.stop_gradient(ref_mask)
        }

    else:
        out = {}

    out.update({
        "alpha": alpha,
        "trans": trans,
        "weights": weights,
        "acc": acc,
        "depth": depth,
        "rgb": rgb,
    })

    return out

def volrend_blending(
    out_static: Dict[str, jnp.ndarray],
    out_dynamic: Dict[str, jnp.ndarray],
    samples: struct.Samples,
    bkgd_rgb: jnp.ndarray,
    use_sample_at_infinity: bool,
    stage: str = 'bri',
    eps: float = 1e-10,
) -> Dict[str, jnp.ndarray]:
    """Render through volume by numerical integration.

    Args:
        out (Dict[str, jnp.ndarray]): A dictionary holding at least
            "point_sigma" (..., S, 1) and "point_rgb" (..., S, 3).
        samples (struct.Samples): Samples to render of shape (..., S).
        bkgd_rgb (jnp.ndarray): Background color of shape (3,).
        use_sample_at_infinity (bool): Whether to sample at infinity.

    Returns:
        Dict[str, jnp.ndarray]: rendering results.
    """
    assert samples.tvals is not None
    assert "point_sigma" in out_dynamic and "point_rgb" in out_dynamic
    assert "point_sigma" in out_static and "point_rgb" in out_static and "point_blending" in out_static
    
    if stage == 'mdd':
        if samples.xs.shape[0] != out_static["point_sigma"].shape[0]:
            blurry_step = int(out_static["point_sigma"].shape[0] / samples.xs.shape[0])
            expand_samples = struct.Samples(
                xs= jnp.broadcast_to(samples.xs[:,None], (samples.xs.shape[0], blurry_step) + samples.xs.shape[1:]).reshape(-1, *samples.xs.shape[1:]),
                directions= jnp.broadcast_to(samples.directions[:,None], (samples.directions.shape[0], blurry_step) + samples.directions.shape[1:]).reshape(-1, *samples.directions.shape[1:]),
                tvals= jnp.broadcast_to(samples.tvals[:,None], (samples.tvals.shape[0], blurry_step) + samples.tvals.shape[1:]).reshape(-1, *samples.tvals.shape[1:])
            )
            samples = expand_samples
    
    batch_shape = samples.xs.shape[:-1]

    # TODO(keunhong): Remove this hack.
    # NOTE(Hang Gao @ 07/15): Actually needed by Nerfies & HyperNeRF to always
    # stop by the scene bound otherwise it will not when trained without depth.
    last_sample_t = 1e10 if use_sample_at_infinity else 1e-19

    # (..., S, 1)
    dists = jnp.concatenate(
        [
            samples.tvals[..., 1:, :] - samples.tvals[..., :-1, :],
            jnp.broadcast_to(jnp.array([last_sample_t]), batch_shape[:-1] + (1, 1)),
        ],
        -2,
    )
    dists = dists * jnp.linalg.norm(samples.directions, axis=-1, keepdims=True)

    raw_blend_w = out_static['point_blending']
    # noise_flag = (out_static["weights"].argmax(-2) <= 5).squeeze(-1).astype(jnp.float32)
    # filter_blending = (1 - noise_flag)[:,None,None]
    # raw_blend_w = (filter_blending)*raw_blend_w + (1-filter_blending)*(1 - raw_blend_w)

    alpha_rig = (1 - jnp.exp(-out_static["point_sigma"] * dists)) * raw_blend_w
    alpha_dy = (1 - jnp.exp(-out_dynamic["point_sigma"] * dists)) * (1 - raw_blend_w)

    trans = jnp.concatenate(
        [
            jnp.ones_like(alpha_dy[..., :1, :], alpha_dy.dtype),
            jnp.cumprod((1 - alpha_rig[..., :-1, :]) * (1 - alpha_dy[..., :-1, :]) + eps, axis=-2),
        ],
        axis=-2,
    )

    weights_st = alpha_rig * trans
    weights_dy = alpha_dy * trans
    weights_mix = weights_dy + weights_st

    # (..., 1)
    acc = (
        weights_mix[..., :-1, :].sum(axis=-2)
        if use_sample_at_infinity
        else weights_mix.sum(axis=-2)
    )

    # (..., 1)
    # Avoid 0/0 case.
    depth = (
        weights_mix[..., :-1, :] * samples.tvals[..., :-1, :]
        if use_sample_at_infinity
        else weights_mix * samples.tvals
    ).sum(axis=-2) / acc.clip(1e-12)

    # This nan_to_num trick from Jon does not really work for the 0/0 case and
    # will cause NaN gradient.
    depth = jnp.clip(
        jnp.nan_to_num(depth, nan=jnp.inf),
        samples.tvals[..., 0, :],
        samples.tvals[..., -1, :],
    )

    # (..., 3)
    rgb = (weights_st * out_static["point_rgb"] + weights_dy * out_dynamic["point_rgb"]).sum(axis=-2)
    rgb = rgb + bkgd_rgb * (1 - acc)
    # weights_map_dd = jax.lax.stop_gradient(jnp.sum(weights_dy, 1))
    raw_weights_map_dd = jnp.sum(weights_dy, 1)

    weights_map_dd = jax.lax.stop_gradient(raw_weights_map_dd)
    weights_map_dd = jnp.where(weights_map_dd > 0.5, 1.0, 0.0)

    out = {
        "weights": weights_mix,
        "depth": depth,
        "rgb": rgb,
        "mask": weights_map_dd,
        "raw_mask": raw_weights_map_dd,
        "point_blending": out_static["point_blending"],
    }
    return out