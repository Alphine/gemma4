# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def colsum_kernel(x_ref, o_ref, *, n_rows, br):
    i = pl.program_id(0)

    @pl.when(i == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    rows = i * br + jnp.arange(br)[:, None]
    x = jnp.where(rows < n_rows, x_ref[...], 0.0)
    o_ref[...] += jnp.sum(x, axis=0, keepdims=True)


def op(x):
    br = 8
    R, C = x.shape
    return pl.pallas_call(
        functools.partial(colsum_kernel, n_rows=R, br=br),
        out_shape=jax.ShapeDtypeStruct((1, C), x.dtype),
        grid=(pl.cdiv(R, br),),
        in_specs=[pl.BlockSpec((br, C), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, C), lambda i: (0, 0)),
        interpret=True,
    )(x)


def reference(x):
    return jnp.sum(x, axis=0, keepdims=True)


def make_cases():
    return [(jax.random.normal(jax.random.key(0), (20, 128)),), (jax.random.normal(jax.random.key(1), (16, 64)),),
            (jax.random.normal(jax.random.key(2), (27, 32)),)]
