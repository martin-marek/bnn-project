import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers


def train_sgd(params, log_likelihood_fn, n_epochs=1_000, lr=1e-5, m=0.9):
    # define negative-log-likelihood loss
    def loss_fn(params):
        return -log_likelihood_fn(params)
    
    # create optimizer
    opt_init, opt_update, get_params = optimizers.momentum(lr, m)
    opt_state = opt_init(params)

    # define update step for each epoch
    def step(i, args):
        opt_state, loss_history = args
        params = get_params(opt_state)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        opt_state = opt_update(i, grads, opt_state)
        loss_history = loss_history.at[i].set(loss)
        return opt_state, loss_history

    # train for 'n_epochs'
    loss_history = jnp.zeros(n_epochs)
    opt_state, loss_history = jax.lax.fori_loop(0, n_epochs, step, (opt_state, loss_history))
        
    # get final parameters
    params_final = get_params(opt_state)
    
    return params_final, loss_history
