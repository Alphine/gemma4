# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def msoftmax_kernel(len_ref, x_ref, o_ref):
    x = x_ref[...]
    valid = jnp.arange(x.shape[-1])[None, :] < len_ref[...]
    x = jnp.where(valid, x, -jnp.inf)
    x = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.where(valid, jnp.exp(x), 0.0)
    o_ref[...] = e / jnp.sum(e, axis=-1, keepdims=True)


def op(lens, x):
    br = 8
    R, C = x.shape
    return pl.pallas_call(
        msoftmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(R // br,),
        in_specs=[pl.BlockSpec((br, 1), lambda i: (i, 0)), pl.BlockSpec((br, C), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((br, C), lambda i: (i, 0)),
        interpret=True,
    )(lens, x)


def reference(lens, x):
    valid = jnp.arange(x.shape[-1])[None, :] < lens
    z = jnp.where(valid, x, -jnp.inf)
    return jnp.where(valid, jax.nn.softmax(z, axis=-1), 0.0)


def make_cases():
    out = []
    for i, (R, C) in enumerate([(16, 64), (24, 48)]):
        lens = jax.random.randint(jax.random.key(i), (R, 1), 1, C + 1).astype(jnp.int32)
        out.append((lens, 2.0 * jax.random.normal(jax.random.key(i + 7), (R, C))))
    return out
