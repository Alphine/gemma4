# Seed kernel (original code, Apache-2.0) for the Pallas bug-injection micro-benchmark.
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def gather_kernel(ids_ref, table_ref, o_ref):
    o_ref[...] = table_ref[...]


def op(ids, table):
    n = ids.shape[0]
    D = table.shape[1]
    return pl.pallas_call(
        gather_kernel,
        out_shape=jax.ShapeDtypeStruct((n, D), table.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(n,),
            in_specs=[pl.BlockSpec((1, D), lambda i, ids_ref: (ids_ref[i], 0))],
            out_specs=pl.BlockSpec((1, D), lambda i, ids_ref: (i, 0)),
        ),
        interpret=True,
    )(ids, table)


def reference(ids, table):
    return table[ids]


def make_cases():
    out = []
    for i, (n, V, D) in enumerate([(12, 20, 16), (7, 11, 8)]):
        out.append((jax.random.randint(jax.random.key(i), (n,), 0, V).astype(jnp.int32),
                    jax.random.normal(jax.random.key(i + 3), (V, D))))
    return out
