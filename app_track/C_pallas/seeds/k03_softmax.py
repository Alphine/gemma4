# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x)
    o_ref[...] = e / jnp.sum(e, axis=-1, keepdims=True)


def op(x):
    br = 8
    R, C = x.shape
    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(R // br,),
        in_specs=[pl.BlockSpec((br, C), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((br, C), lambda i: (i, 0)),
        interpret=True,
    )(x)


def reference(x):
    return jax.nn.softmax(x, axis=-1)


def make_cases():
    return [(3.0 * jax.random.normal(jax.random.key(0), (32, 64)),), (3.0 * jax.random.normal(jax.random.key(1), (24, 40)),)]
