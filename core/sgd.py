import jax
import jax.numpy as jnp


def train_sgd(params, log_likelihood_fn, n_epochs=100_000, lr_start=1e-3, lr_stop=1e-8, m=0):
    # define negative-log-likelihood loss
    def loss_fn(params):
        return -log_likelihood_fn(params)
    
    # calculate lr decay - using an exponential schedule
    lr_decay = (lr_stop/lr_start)**(1/n_epochs)

    # define update step for each epoch
    def step(i, args):
        params, loss_history = args
        lr = lr_start*lr_decay**i
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params = jax.tree_multimap(lambda x, g: x - lr*g, params, grads)
        loss_history = loss_history.at[i].set(loss)
        return params, loss_history

    # train for 'n_epochs'
    loss_history = jnp.zeros(n_epochs)
    params, loss_history = jax.lax.fori_loop(0, n_epochs, step, (params, loss_history))
    
    return params, loss_history
