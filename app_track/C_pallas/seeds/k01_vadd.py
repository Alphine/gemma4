# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def op(x, y):
    bn = 64
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(pl.cdiv(x.shape[0], bn),),
        in_specs=[pl.BlockSpec((bn,), lambda i: (i,)), pl.BlockSpec((bn,), lambda i: (i,))],
        out_specs=pl.BlockSpec((bn,), lambda i: (i,)),
        interpret=True,
    )(x, y)


def reference(x, y):
    return x + y


def make_cases():
    cases = []
    for n in (256, 200):
        cases.append((jax.random.normal(jax.random.key(n), (n,)), jax.random.normal(jax.random.key(n + 1), (n,))))
    return cases
