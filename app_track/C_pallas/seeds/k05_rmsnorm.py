# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def rmsnorm_kernel(x_ref, w_ref, o_ref, *, eps):
    x = x_ref[...].astype(jnp.float32)
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    o_ref[...] = (x * jax.lax.rsqrt(ms + eps) * w_ref[...]).astype(o_ref.dtype)


def op(x, w):
    br = 4
    R, C = x.shape
    return pl.pallas_call(
        functools.partial(rmsnorm_kernel, eps=1e-6),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(R // br,),
        in_specs=[pl.BlockSpec((br, C), lambda i: (i, 0)), pl.BlockSpec((1, C), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((br, C), lambda i: (i, 0)),
        interpret=True,
    )(x, w)


def reference(x, w):
    return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w


def make_cases():
    return [
        (jax.random.normal(jax.random.key(0), (16, 64)), 1.0 + jax.random.normal(jax.random.key(1), (1, 64))),
        (jax.random.normal(jax.random.key(2), (8, 96)), 1.0 + jax.random.normal(jax.random.key(3), (1, 96))),
    ]
