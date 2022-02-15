import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
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


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def normal_like_tree(a, key):
    treedef = jax.tree_structure(a)
    num_vars = len(jax.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_multimap(lambda p, k: jax.random.normal(k, shape=p.shape), a, jax.tree_unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]


def update_step_size(step_size, accept_prob, target_accept_rate=0, step_size_adaptation_speed=10):
    if target_accept_rate > 0 and step_size_adaptation_speed > 0:
        step_size *= jnp.exp(step_size_adaptation_speed * (accept_prob - target_accept_rate))
    return step_size
   

def rwmh_sampler(params, log_prob_fn, key, n_steps=100, n_blind_steps=100, step_size=1e-4, target_accept_rate=0.23, step_size_adaptation_speed=0.1):
    
    # define a step that doesn't keep history
    def step_without_history(i, args):
        params, log_prob, step_size, total_accept_prob, key = args
        key, normal_key, uniform_key = jax.random.split(key, 3)
        
        # propose new parameters
        step, _ = normal_like_tree(params, normal_key)
        params_new = jax.tree_multimap(lambda a, b: a + step_size * b, params, step)
        
        # decide whether to accept new position
        log_prob_new = log_prob_fn(params_new)
        log_accept_prob = log_prob_new - log_prob
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        total_accept_prob += accept_prob
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, params_new, params)
        log_prob = ifelse(accept, log_prob_new, log_prob)
        
        # update step size
        step_size = update_step_size(step_size, accept_prob, target_accept_rate, step_size_adaptation_speed)
        
        return params, log_prob, step_size, total_accept_prob, key
    
    # define a step that keeps history
    def step_with_history(i, args):
        params, params_history, step_size, total_accept_prob, key = args
        
        # do 'n_blind_steps', without keeping history
        log_prob = log_prob_fn(params)
        params, log_prob, step_size, inner_total_accept_prob, key = jax.lax.fori_loop(0, n_blind_steps, step_without_history, (params, log_prob, step_size, 0, key))
        total_accept_prob += inner_total_accept_prob / n_blind_steps
        
        # store history
        params_raveled, _ = ravel_pytree(params)
        params_history = params_history.at[0].set(params_raveled)
              
        return params, params_history, step_size, total_accept_prob, key
    
    # ravel params
    params_raveled, unravel_fn = ravel_pytree(params)
    
    # do 'n_steps'
    params_history_raveled = jnp.zeros([n_steps]+list(params_raveled.shape))
    params_history_raveled = params_history_raveled.at[0].set(params_raveled)
    _, params_history_raveled, step_size, total_accept_prob, key = jax.lax.fori_loop(1, n_steps, step_with_history, (params, params_history_raveled, step_size, 0, key))
    
    # unravel params
    params_history_unraveled = [unravel_fn(params_raveled) for params_raveled in params_history_raveled]
    
    print(f'Avg. accept. prob.: {(total_accept_prob/n_steps):.2%}')
    return params_history_unraveled


def leapfrog(params, momentum, log_prob_fn, step_size, n_steps):
    
    # define a single step
    def step(i, args):
        params, momentum = args
        
        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g, momentum, grad)

        # update params
        params = jax.tree_multimap(lambda s, m: s + m * step_size, params, momentum)

        # update momentum
        grad = jax.grad(log_prob_fn)(params)
        momentum = jax.tree_multimap(lambda m, g: m + 0.5 * step_size * g, momentum, grad)
        
        return params, momentum

    # do 'n_steps'
    new_params, new_momentum = jax.lax.fori_loop(0, n_steps, step, (params, momentum))

    return new_params, new_momentum


def hmc_sampler(params, log_prob_fn, n_steps, n_leapfrog_steps, step_size, key, target_accept_rate=0.8, step_size_adaptation_speed=1):

    # define a single step
    def step(i, args):
        params, params_history, step_size, total_accept_prob, key = args
        key, normal_key, uniform_key = jax.random.split(key, 3)

        # generate random momentum
        momentum, _ = normal_like_tree(params, normal_key)

        # leapfrog
        new_params, new_momentum = leapfrog(params, momentum, log_prob_fn, step_size, n_leapfrog_steps)

        # MH correction
        potentaial_energy_diff = log_prob_fn(new_params) - log_prob_fn(params)
        kinetic_energy_diff = 0.5*sum([jnp.sum(m1**2-m2**2) for m1, m2 in zip(jax.tree_leaves(momentum), jax.tree_leaves(new_momentum))])
        log_accept_prob = potentaial_energy_diff + kinetic_energy_diff
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        total_accept_prob += accept_prob
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = ifelse(accept, new_params, params)
        
        # store history
        params_raveled, _ = ravel_pytree(params)
        params_history = params_history.at[0].set(params_raveled)
        
        # update step size
        step_size = update_step_size(step_size, accept_prob, target_accept_rate, step_size_adaptation_speed)
        
        return params, params_history, step_size, total_accept_prob, key
    
    # ravel params
    params_raveled, unravel_fn = ravel_pytree(params)
    
    # do 'n_steps'
    params_history_raveled = jnp.zeros([n_steps]+list(params_raveled.shape))
    params_history_raveled = params_history_raveled.at[0].set(params_raveled)
    _, params_history_raveled, step_size, total_accept_prob, key = jax.lax.fori_loop(1, n_steps, step, (params, params_history_raveled, step_size, 0, key))
    
    # unravel params
    params_history_unraveled = [unravel_fn(params_raveled) for params_raveled in params_history_raveled]
    
    print(f'Avg. accept. prob.: {(total_accept_prob/n_steps):.2%}')
    return params_history_unraveled
