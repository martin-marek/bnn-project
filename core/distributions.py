import jax
import jax.numpy as jnp


def gaussian_log_pdf(y, mean, std):
    return -0.5 * jnp.log(std**2 * 2 * jnp.pi) - 0.5 * ((y-mean)/std)**2


def make_gaussian_log_likelihood(x, y, predict_fn):
    
    def out_fn(params, x, y):
        # predict y
        y_hat = predict_fn(x, params)
        mean = y_hat[:, 0]
        std = y_hat[:, 1]

        # compute likelihood
        pointwise_likelihood = gaussian_log_pdf(y, mean, std)
        total_likelihood = jnp.sum(pointwise_likelihood)
        
        return total_likelihood
    
    return out_fn


def make_gaussian_log_prior(std):
    
    def log_prior(params):
        n_params = sum([p.size for p in jax.tree_leaves(params)])
        dy2 = sum(jax.tree_leaves((jax.tree_map(lambda x: jnp.sum(x**2), params))))
        log_prob = -0.5 * n_params * jnp.log(std**2 * 2 * jnp.pi) - 0.5 * dy2/std**2
        return log_prob
    
    return log_prior
