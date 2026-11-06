# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def transpose_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...].T


def op(x):
    bm, bn = 16, 32
    M, N = x.shape
    return pl.pallas_call(
        transpose_kernel,
        out_shape=jax.ShapeDtypeStruct((N, M), x.dtype),
        grid=(M // bm, N // bn),
        in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((bn, bm), lambda i, j: (j, i)),
        interpret=True,
    )(x)


def reference(x):
    return x.T


def make_cases():
    return [(jax.random.normal(jax.random.key(0), (48, 64)),), (jax.random.normal(jax.random.key(1), (32, 96)),)]
