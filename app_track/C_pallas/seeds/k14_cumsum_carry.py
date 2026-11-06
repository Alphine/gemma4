# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def cumsum_kernel(x_ref, o_ref, carry_ref):
    i = pl.program_id(0)

    @pl.when(i == 0)
    def _init():
        carry_ref[...] = jnp.zeros_like(carry_ref)

    local = jnp.cumsum(x_ref[...], axis=0)
    o_ref[...] = local + carry_ref[...]
    carry_ref[...] = carry_ref[...] + jnp.sum(x_ref[...], axis=0, keepdims=True)


def op(x):
    S, D = x.shape
    bs = 8
    out, _ = pl.pallas_call(
        cumsum_kernel,
        out_shape=(jax.ShapeDtypeStruct((S, D), x.dtype), jax.ShapeDtypeStruct((1, D), x.dtype)),
        grid=(S // bs,),
        in_specs=[pl.BlockSpec((bs, D), lambda i: (i, 0))],
        out_specs=(pl.BlockSpec((bs, D), lambda i: (i, 0)), pl.BlockSpec((1, D), lambda i: (0, 0))),
        interpret=True,
    )(x)
    return out


def reference(x):
    return jnp.cumsum(x, axis=0)


def make_cases():
    return [(jax.random.normal(jax.random.key(0), (32, 16)),), (jax.random.normal(jax.random.key(1), (24, 8)),)]
