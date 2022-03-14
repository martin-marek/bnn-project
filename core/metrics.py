import jax
import jax.numpy as jnp


def autocorr(y, n_lags=20):
    """
    y.shape: [num_chains (M) x num_steps (N) x num_features (B)]
    """
    M, N, B = y.shape

    # a single step in the loop compute autocovariance for a given lag
    def step(i, r):
        a = y[:, n_lags:, :]
        b = jnp.roll(y, i, axis=1)[:, n_lags:, :]
        cov_ab = ((a - a.mean(axis=(0, 1), keepdims=True)) * (b - b.mean(axis=(0, 1), keepdims=True))).mean(axis=1)
        var_a = ((a - a.mean(axis=(0, 1), keepdims=True))**2).mean(axis=1)
        var_b = ((b - a.mean(axis=(0, 1), keepdims=True))**2).mean(axis=1)
        cor = cov_ab / jnp.sqrt(var_a*var_b)
        r = r.at[:, i, :].set(cor)
        return r

    # compute autocovariance for each chain and feature separately
    r = jnp.zeros([M, n_lags, B])
    r = r.at[:, 0, :].set(1)
    r = jax.lax.fori_loop(1, n_lags, step, r)

    # take a mean along chains and features
    r = r.mean(axis=[0, 2])
    
    return r


def r_hat(y):
    """
    Based on the paper "What Are Bayesian Neural Network Posteriors Really Like? - Appendix B"
    y.shape: [num_chains (M) x num_steps (N) x num_features (B)]
    """
    M, N, B = y.shape

    # compute variances
    b = (N/(M-1)) * ((y.mean(axis=1, keepdims=True) - y.mean(axis=(0, 1), keepdims=True))**2).sum(axis=(0, 1))
    w = 1/(M*(N-1)) * ((y - y.mean(axis=1, keepdims=True))**2).sum(axis=(0, 1))
    r_hat = jnp.sqrt((((N-1)/N)*w + (1/N)*b)/w)
    
    return r_hat