import jax
import jax.numpy as jnp
from functools import partial


def split(x, n_batches):
    batch_size = len(x) // n_batches
    x_batched = x[:(n_batches*batch_size)].reshape([n_batches, batch_size, *x.shape[1:]])
    return x_batched


def pmap_(f):
    def out_fn(x, y, params, log_likelihood_fn, log_prior_fn, n_dev=1, *args):
        x_batched = split(x, n_dev)
        y_batched = split(y, n_dev)
        params_batched = jnp.repeat(params[None], n_dev, axis=0)
        @partial(jax.pmap, axis_name='batch')
        def g(x, y, params):
            log_posterior_fn = make_log_posterior_fn(x, y, log_likelihood_fn, log_prior_fn)
            return f(log_posterior_fn, params, *args)
        out_batched = g(x_batched, y_batched, params_batched)
        out_single = [out[0] for out in out_batched]
        return out_single
    return out_fn


def make_log_posterior_fn(x, y, log_likelihood_fn, log_prior_fn):
    def out_fn(params):
        log_likelihood = log_likelihood_fn(params, x, y)
        log_likelihood = jax.lax.psum(log_likelihood, axis_name='batch')
        log_prior = log_prior_fn(params)
        return log_likelihood + log_prior
    return out_fn
