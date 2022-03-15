import jax
import jax.numpy as jnp
from functools import partial

def split(x, n_batches):
    batch_size = len(x) // n_batches
    x_batched = x[:(n_batches*batch_size)].reshape([n_batches, batch_size, *x.shape[1:]])
    return x_batched

def pmap_(f, x, y, params, n_dev=1, *args):
    x_batched = split(x, n_dev)
    y_batched = split(y, n_dev)
    params_batched = jnp.repeat(params[None], n_dev, axis=0)
    @partial(jax.pmap, axis_name='batch')
    def g(x, y, params):
        return f(x, y, params, *args)
    out_batched = g(x_batched, y_batched, params_batched)
    out_single = [out[0] for out in out_batched]
    return out_single
