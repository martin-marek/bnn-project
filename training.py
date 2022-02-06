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


def normal_like_tree(a, key):
    treedef = jax.tree_structure(a)
    num_vars = len(jax.tree_leaves(a))
    all_keys = jax.random.split(key, num=(num_vars + 1))
    noise = jax.tree_multimap(lambda p, k: jax.random.normal(k, shape=p.shape), a, jax.tree_unflatten(treedef, all_keys[1:]))
    return noise, all_keys[0]


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def rwmh_sampler(params, log_prob_fn, key, n_steps=100, n_blind_steps=100, step_size=1e-4):
    log_prob = log_prob_fn(params)
    params_history = [params]
    log_prob_history = [log_prob]
    
    def step(i, args):
        params, log_prob, key = args
        key, normal_key, uniform_key = jax.random.split(key, 3)
        
        # propose new parameters
        step, _ = normal_like_tree(params, normal_key)
        params_new = jax.tree_multimap(lambda a, b: a + step_size * b, params, step)
        
        # decide whether to accept new position
        log_prob_new = log_prob_fn(params_new)
        log_accept_prob = log_prob_new - log_prob
        accept_prob = jnp.minimum(1, jnp.exp(log_accept_prob))
        accept = jax.random.uniform(uniform_key) < accept_prob
        
        # update current position
        params = ifelse(accept, params_new, params)
        log_prob = ifelse(accept, log_prob_new, log_prob)
            
        return params, log_prob, key
    
    # do large non-vectorized steps, keeping intermediate state
    for i in range(n_steps):
        
        # do small vectorized steps, discarding intermediate state
        params, log_prob, key = jax.lax.fori_loop(0, n_blind_steps, step, (params, log_prob, key))
        
        # store current params
        params_history.append(params)
        log_prob_history.append(log_prob)
    
    return params_history, jnp.array(log_prob_history)


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


def hmc_sampler(params, log_prob_fn, n_steps, n_leapfrog_steps, step_size, key, target_accept_rate=0, step_size_adaptation_speed=0):
    params_history = [params]
    log_prob_history = [log_prob_fn(params)]

    for i in range(n_steps):
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
        print(accept_prob)
        if jax.random.uniform(uniform_key) < accept_prob:
            params = new_params
        
        # update step size
        if target_accept_rate > 0 and step_size_adaptation_speed > 0:
            scale_factor = jnp.exp(step_size_adaptation_speed * (accept_prob - target_accept_rate))
            step_size *= scale_factor
        
        # store current params
        params_history.append(params)
        log_prob_history.append(log_prob_fn(params))
    
    return params_history, jnp.array(log_prob_history)
