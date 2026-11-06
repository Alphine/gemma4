# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def l2norm_kernel(x_ref, r_ref, o_ref):
    y = x_ref[...] + r_ref[...]
    n = jnp.sqrt(jnp.sum(y * y, axis=-1, keepdims=True))
    o_ref[...] = y / (n + 1e-6)


def op(x, r):
    R, C = x.shape
    br = 8
    return pl.pallas_call(
        l2norm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(R // br,),
        in_specs=[pl.BlockSpec((br, C), lambda i: (i, 0)), pl.BlockSpec((br, C), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((br, C), lambda i: (i, 0)),
        interpret=True,
    )(x, r)


def reference(x, r):
    y = x + r
    return y / (jnp.sqrt(jnp.sum(y * y, axis=-1, keepdims=True)) + 1e-6)


def make_cases():
    return [(jax.random.normal(jax.random.key(0), (16, 64)), jax.random.normal(jax.random.key(1), (16, 64))),
            (jax.random.normal(jax.random.key(2), (24, 32)), jax.random.normal(jax.random.key(3), (24, 32)))]
