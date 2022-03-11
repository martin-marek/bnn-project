import jax
import jax.numpy as jnp
from . import utils
from .hmc import hmc_sampler
from .nuts import nuts
from .rwmh import rwmh_sampler
from .sgd import train_sgd
from .utils import ravel_pytree_ as ravel_fn


def chain_to_arr(chain):
   return jnp.array([ravel_fn(node) for node in chain])


def initialize_params(key, params, sd):
    # resamples initial model paprameters
    params = utils.normal_like_tree(params, key)
    params = jax.tree_map(lambda x: sd*x, params)
    return params


def vmap_over_keys(f, key, n):
    keys = jnp.array(jax.random.split(key, n))
    return jax.vmap(f)(keys)


def create_sgd_chains(key, log_likelihood_fn, params_init, init_sd, n_epochs, ll_start, ll_stop, n_chains, n_samples):
    def create_chain(key):
        init_key, mcmc_key = jax.random.split(key, 2)
        params = initialize_params(init_key, params_init, init_sd)
        params, loss_history = train_sgd(params, log_likelihood_fn, n_epochs, ll_start, ll_stop)
        return jnp.concatenate([ravel_fn(params), loss_history])

    n_params = len(ravel_fn(params_init))
    out = vmap_over_keys(create_chain, key, n_chains)
    chains = out[:, None, :n_params]
    loss = out[:, n_params:]
    return chains, loss


def create_rwmh_chains(key, log_likelihood_fn, params_init, init_sd, n_epochs, sgd_ll_start, sgd_ll_stop, step_size, n_blind_steps, n_chains, n_samples):
    def create_chain(key):
        init_key, mcmc_key = jax.random.split(key, 2)
        params = initialize_params(init_key, params_init, init_sd)
        params, _ = train_sgd(params, log_likelihood_fn, n_epochs, sgd_ll_start, sgd_ll_stop)
        chain, avg_accept_prob = rwmh_sampler(params, log_likelihood_fn, mcmc_key, n_samples, n_blind_steps, step_size)
        return jnp.concatenate([chain_to_arr(chain).flatten(), jnp.array([avg_accept_prob])])

    n_params = len(ravel_fn(params_init))
    out = vmap_over_keys(create_chain, key, n_chains)
    chains = out[:, :-1].reshape([n_chains, n_samples, -1])
    avg_accept_prob = out[:, -1].mean()
    return chains, avg_accept_prob


def create_hmc_chains(key, log_likelihood_fn, params_init, init_sd, n_epochs, sgd_ll_start, sgd_ll_stop, step_size, n_leapfrog_steps, n_chains, n_samples):
    def create_chain(key):
        init_key, mcmc_key = jax.random.split(key, 2)
        params = initialize_params(init_key, params_init, init_sd)
        params, _ = train_sgd(params, log_likelihood_fn, n_epochs, sgd_ll_start, sgd_ll_stop)
        chain, avg_accept_prob = hmc_sampler(params, log_likelihood_fn, mcmc_key, n_samples, n_leapfrog_steps, step_size)
        return jnp.concatenate([chain_to_arr(chain).flatten(), jnp.array([avg_accept_prob])])

    n_params = len(ravel_fn(params_init))
    out = vmap_over_keys(create_chain, key, n_chains)
    chains = out[:, :-1].reshape([n_chains, n_samples, -1])
    avg_accept_prob = out[:, -1].mean()
    return chains, avg_accept_prob


def create_mixed_chains(key, log_likelihood_fn, params_init, init_sd, n_epochs, sgd_ll_start, sgd_ll_stop, step_size, n_leapfrog_steps, n_chains, n_outer_steps, n_inner_steps):
    chains, avg_accept_prob = create_hmc_chains(key, log_likelihood_fn, params_init, init_sd, n_epochs, sgd_ll_start, sgd_ll_stop, step_size, n_leapfrog_steps, n_chains*n_outer_steps, n_inner_steps)
    chains = chains.reshape([n_chains, n_outer_steps*n_inner_steps, -1])
    return chains, avg_accept_prob

