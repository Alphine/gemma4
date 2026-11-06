# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def matmul_kernel(x_ref, y_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)


def op(x, y, bm=16, bn=32, bk=32):
    M, K = x.shape
    N = y.shape[1]
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // bm, N // bn, K // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        interpret=True,
    )(x, y)


def reference(x, y):
    return x @ y


def make_cases():
    out = []
    for i, (M, K, N) in enumerate([(48, 96, 64), (32, 64, 96)]):
        out.append((jax.random.normal(jax.random.key(i), (M, K)), jax.random.normal(jax.random.key(i + 10), (K, N))))
    return out
