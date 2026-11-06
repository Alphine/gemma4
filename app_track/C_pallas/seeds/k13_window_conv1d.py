# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def conv_kernel(x_ref, w_ref, o_ref, *, bo, width):
    i = pl.program_id(0)
    acc = jnp.zeros((bo,), dtype=jnp.float32)
    for t in range(width):
        acc = acc + x_ref[pl.ds(i * bo + t, bo)] * w_ref[t]
    o_ref[...] = acc


def op(x, w):
    width = w.shape[0]
    n_out = x.shape[0] - width + 1
    bo = 32
    return pl.pallas_call(
        functools.partial(conv_kernel, bo=bo, width=width),
        out_shape=jax.ShapeDtypeStruct((n_out,), jnp.float32),
        grid=(n_out // bo,),
        in_specs=[pl.BlockSpec(x.shape, lambda i: (0,)), pl.BlockSpec(w.shape, lambda i: (0,))],
        out_specs=pl.BlockSpec((bo,), lambda i: (i,)),
        interpret=True,
    )(x, w)


def reference(x, w):
    width = w.shape[0]
    n_out = x.shape[0] - width + 1
    return sum(x[t:t + n_out] * w[t] for t in range(width))


def make_cases():
    return [(jax.random.normal(jax.random.key(0), (131,)), jax.random.normal(jax.random.key(1), (4,))),
            (jax.random.normal(jax.random.key(2), (99,)), jax.random.normal(jax.random.key(3), (4,)))]
