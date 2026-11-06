# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def attn_kernel(q_ref, k_ref, v_ref, o_ref, *, sm_scale):
    s = jnp.dot(q_ref[...], k_ref[...].T) * sm_scale
    s = s - jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s)
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    o_ref[...] = jnp.dot(p, v_ref[...])


def op(q, k, v):
    S, D = q.shape
    bq = 8
    return pl.pallas_call(
        functools.partial(attn_kernel, sm_scale=1.0 / (D ** 0.5)),
        out_shape=jax.ShapeDtypeStruct((S, D), jnp.float32),
        grid=(S // bq,),
        in_specs=[
            pl.BlockSpec((bq, D), lambda i: (i, 0)),
            pl.BlockSpec((k.shape[0], D), lambda i: (0, 0)),
            pl.BlockSpec((v.shape[0], D), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((bq, D), lambda i: (i, 0)),
        interpret=True,
    )(q, k, v)


def reference(q, k, v):
    D = q.shape[1]
    return jax.nn.softmax(q @ k.T / (D ** 0.5), axis=-1) @ v


def make_cases():
    out = []
    for i, (S, T, D) in enumerate([(32, 32, 16), (16, 48, 32)]):
        out.append(tuple(jax.random.normal(jax.random.key(10 * i + j), (n, D)) for j, n in enumerate((S, T, T))))
    return out
