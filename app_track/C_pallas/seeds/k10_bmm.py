# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def bmm_kernel(x_ref, y_ref, o_ref):
    o_ref[0] = jnp.dot(x_ref[0], y_ref[0], preferred_element_type=jnp.float32)


def op(x, y):
    B, M, K = x.shape
    N = y.shape[2]
    bm, bn = 16, 16
    return pl.pallas_call(
        bmm_kernel,
        out_shape=jax.ShapeDtypeStruct((B, M, N), jnp.float32),
        grid=(B, M // bm, N // bn),
        in_specs=[
            pl.BlockSpec((1, bm, K), lambda b, i, j: (b, i, 0)),
            pl.BlockSpec((1, K, bn), lambda b, i, j: (b, 0, j)),
        ],
        out_specs=pl.BlockSpec((1, bm, bn), lambda b, i, j: (b, i, j)),
        interpret=True,
    )(x, y)


def reference(x, y):
    return jnp.einsum("bmk,bkn->bmn", x, y)


def make_cases():
    return [
        (jax.random.normal(jax.random.key(0), (3, 32, 24)), jax.random.normal(jax.random.key(1), (3, 24, 48))),
        (jax.random.normal(jax.random.key(2), (2, 16, 40)), jax.random.normal(jax.random.key(3), (2, 40, 32))),
    ]
