import jax
import jax.numpy as jnp
from functools import partial
from .spmd import pmap_

def train_sgd(x, y, params, log_likelihood_fn, log_prior_fn, n_epochs, lr_start, lr_stop):
    
    # define negative-log-likelihood loss
    def loss_fn(params, x, y):
        log_posterior = log_prior_fn(params) + log_likelihood_fn(params, x, y)
        return -log_posterior
    
    # calculate lr decay - using an exponential schedule
    lr_decay = (lr_stop/lr_start)**(1/n_epochs)

    # define update step for each epoch
    def step(i, args):
        params, loss_history = args
        lr = lr_start*lr_decay**i
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        loss = jax.lax.psum(loss, axis_name='batch')
        grads = jax.lax.psum(grads, axis_name='batch')
        params = jax.tree_multimap(lambda x, g: x - lr*g, params, grads)
        loss_history = loss_history.at[i].set(loss)
        return params, loss_history

    # train for 'n_epochs'
    loss_history = jnp.zeros(n_epochs)
    params, loss_history = jax.lax.fori_loop(0, n_epochs, step, (params, loss_history))
    
    return params, loss_history


def train_ensamble(x, y, params_init, *args):
    @partial(jax.vmap)
    def train_in_parallel(params):
        return train_sgd(x, y, params, *args)
    params, loss_history = train_in_parallel(params_init)
    params = params.reshape([-1, params.shape[-1]])
    loss_history = loss_history.reshape([-1, loss_history.shape[-1]])
    return params,  loss_history

