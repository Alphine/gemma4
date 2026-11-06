# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def rmatmul_kernel(x_ref, y_ref, o_ref, *, K, bk):
    kk = pl.program_id(2)

    @pl.when(kk == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)

    cols = kk * bk + jnp.arange(bk)
    x = jnp.where(cols[None, :] < K, x_ref[...], 0.0)
    y = jnp.where(cols[:, None] < K, y_ref[...], 0.0)
    o_ref[...] += jnp.dot(x, y, preferred_element_type=jnp.float32)


def op(x, y):
    M, K = x.shape
    N = y.shape[1]
    bm, bn, bk = 16, 32, 32
    return pl.pallas_call(
        functools.partial(rmatmul_kernel, K=K, bk=bk),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // bm, N // bn, pl.cdiv(K, bk)),
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
    for i, (M, K, N) in enumerate([(32, 80, 64), (16, 50, 32)]):
        out.append((jax.random.normal(jax.random.key(i), (M, K)), jax.random.normal(jax.random.key(i + 10), (K, N))))
    return out
