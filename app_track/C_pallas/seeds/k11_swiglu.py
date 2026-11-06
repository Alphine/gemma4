# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def swiglu_kernel(x_ref, wg_ref, wu_ref, o_ref):
    x = x_ref[...]
    g = jnp.dot(x, wg_ref[...], preferred_element_type=jnp.float32)
    u = jnp.dot(x, wu_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = (g * jax.nn.sigmoid(g)) * u


def op(x, wg, wu):
    M, K = x.shape
    H = wg.shape[1]
    bm, bh = 16, 32
    return pl.pallas_call(
        swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((M, H), jnp.float32),
        grid=(M // bm, H // bh),
        in_specs=[
            pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, bh), lambda i, j: (0, j)),
            pl.BlockSpec((K, bh), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bh), lambda i, j: (i, j)),
        interpret=True,
    )(x, wg, wu)


def reference(x, wg, wu):
    g = x @ wg
    return (g * jax.nn.sigmoid(g)) * (x @ wu)


def make_cases():
    out = []
    for i, (M, K, H) in enumerate([(32, 24, 64), (16, 40, 96)]):
        out.append((jax.random.normal(jax.random.key(3 * i), (M, K)),
                    0.3 * jax.random.normal(jax.random.key(3 * i + 1), (K, H)),
                    0.3 * jax.random.normal(jax.random.key(3 * i + 2), (K, H))))
    return out
