# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def flash_kernel(q_ref, k_ref, v_ref, o_ref, *, bk, sm_scale, causal):
    bq, d = q_ref.shape
    qi = pl.program_id(0)
    q = q_ref[...] * sm_scale
    m0 = jnp.full((bq,), -jnp.inf, dtype=jnp.float32)
    l0 = jnp.zeros((bq,), dtype=jnp.float32)
    a0 = jnp.zeros((bq, d), dtype=jnp.float32)

    def body(j, carry):
        m, l, acc = carry
        kb = k_ref[pl.ds(j * bk, bk), :]
        vb = v_ref[pl.ds(j * bk, bk), :]
        s = jnp.dot(q, kb.T)
        if causal:
            rows = qi * bq + jnp.arange(bq)[:, None]
            cols = j * bk + jnp.arange(bk)[None, :]
            s = jnp.where(rows >= cols, s, -1e30)
        m_new = jnp.maximum(m, jnp.max(s, axis=-1))
        p = jnp.exp(s - m_new[:, None])
        corr = jnp.exp(m - m_new)
        l = l * corr + jnp.sum(p, axis=-1)
        acc = acc * corr[:, None] + jnp.dot(p, vb)
        return m_new, l, acc

    m, l, acc = jax.lax.fori_loop(0, k_ref.shape[0] // bk, body, (m0, l0, a0))
    o_ref[...] = acc / l[:, None]


def op(q, k, v, causal=False):
    S, D = q.shape
    bq, bk = 16, 16
    return pl.pallas_call(
        functools.partial(flash_kernel, bk=bk, sm_scale=1.0 / (D ** 0.5), causal=causal),
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


def reference(q, k, v, causal=False):
    S, D = q.shape
    s = q @ k.T / (D ** 0.5)
    if causal:
        s = jnp.where(jnp.arange(S)[:, None] >= jnp.arange(k.shape[0])[None, :], s, -1e30)
    return jax.nn.softmax(s, axis=-1) @ v


def make_cases():
    out = []
    for causal in (False, True):
        for i, (S, D) in enumerate([(64, 16), (32, 32)]):
            out.append(tuple(jax.random.normal(jax.random.key(7 * i + j), (S, D)) for j in range(3)) + (causal,))
    return out
