# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def layernorm_kernel(x_ref, g_ref, b_ref, o_ref, *, eps):
    x = x_ref[...]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    xc = x - mean
    var = jnp.mean(xc * xc, axis=-1, keepdims=True)
    o_ref[...] = xc * jax.lax.rsqrt(var + eps) * g_ref[...] + b_ref[...]


def op(x, g, b):
    br = 8
    R, C = x.shape
    return pl.pallas_call(
        functools.partial(layernorm_kernel, eps=1e-5),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(R // br,),
        in_specs=[
            pl.BlockSpec((br, C), lambda i: (i, 0)),
            pl.BlockSpec((1, C), lambda i: (0, 0)),
            pl.BlockSpec((1, C), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((br, C), lambda i: (i, 0)),
        interpret=True,
    )(x, g, b)


def reference(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g + b


def make_cases():
    out = []
    for i, (R, C) in enumerate([(32, 64), (16, 128)]):
        x = 2.0 + jax.random.normal(jax.random.key(i), (R, C))
        g = 1.0 + 0.5 * jax.random.normal(jax.random.key(i + 5), (1, C))
        b = 0.3 * jax.random.normal(jax.random.key(i + 9), (1, C))
        out.append((x, g, b))
    return out
