# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def rope_kernel(x_ref, cos_ref, sin_ref, o_ref):
    x = x_ref[...]
    half = x.shape[-1] // 2
    x1 = x[:, :half]
    x2 = x[:, half:]
    c = cos_ref[...]
    s = sin_ref[...]
    o_ref[:, :half] = x1 * c - x2 * s
    o_ref[:, half:] = x2 * c + x1 * s


def op(x, cos, sin):
    S, D = x.shape
    bs = 8
    return pl.pallas_call(
        rope_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(S // bs,),
        in_specs=[
            pl.BlockSpec((bs, D), lambda i: (i, 0)),
            pl.BlockSpec((bs, D // 2), lambda i: (i, 0)),
            pl.BlockSpec((bs, D // 2), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((bs, D), lambda i: (i, 0)),
        interpret=True,
    )(x, cos, sin)


def reference(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[:, :half], x[:, half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_cases():
    out = []
    for i, (S, D) in enumerate([(32, 16), (24, 32)]):
        ang = jax.random.uniform(jax.random.key(i), (S, D // 2), minval=0.0, maxval=3.0)
        out.append((jax.random.normal(jax.random.key(i + 4), (S, D)), jnp.cos(ang), jnp.sin(ang)))
    return out
