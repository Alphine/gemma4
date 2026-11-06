# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def tri_kernel(a_ref, b_ref, o_ref, *, bm, bk):
    i = pl.program_id(0)
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    @pl.when(k * bk <= i * bm + bm - 1)
    def _acc():
        rows = i * bm + jnp.arange(bm)[:, None]
        cols = k * bk + jnp.arange(bk)[None, :]
        a = jnp.where(rows >= cols, a_ref[...], 0.0)
        o_ref[...] += jnp.dot(a, b_ref[...], preferred_element_type=jnp.float32)


def op(a, b):
    M, K = a.shape
    N = b.shape[1]
    bm, bn, bk = 16, 16, 16
    return pl.pallas_call(
        functools.partial(tri_kernel, bm=bm, bk=bk),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // bm, N // bn, K // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        interpret=True,
    )(a, b)


def reference(a, b):
    return jnp.tril(a) @ b


def make_cases():
    return [(jax.random.normal(jax.random.key(0), (64, 64)), jax.random.normal(jax.random.key(1), (64, 32))),
            (jax.random.normal(jax.random.key(2), (48, 48)), jax.random.normal(jax.random.key(3), (48, 48)))]
