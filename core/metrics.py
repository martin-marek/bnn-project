import jax
import jax.numpy as jnp


def autocorr(chains, f=None, n_lags=20):
    """
    chains: [num_chains x num_steps x num_params]
    f: f(chain_node) -> [B]: transforms params to a 1D quantity of interest, eg predictive distribution
    """
    
    # map model parameters to a 1D array of features
    y = jnp.array([[f(params) for params in chain] for chain in chains])
    M, N, B = y.shape

    # a single step in the loop compute autocovariance for a given lag
    def step(i, r):
        a = y[:, n_lags:, :]
        b = jnp.roll(y, i, axis=1)[:, n_lags:, :]
        cov = ((a - a.mean(axis=1, keepdims=True)) * (b - b.mean(axis=1, keepdims=True))).mean(axis=1)
        a_std = jnp.sqrt((a**2).mean(axis=1) - a.mean(axis=1)**2)
        b_std = jnp.sqrt((b**2).mean(axis=1) - b.mean(axis=1)**2)
        cor = cov / (a_std*b_std)
        r = r.at[:, i, :].set(cor)
        return r

    # compute autocovariance for each chain and feature separately
    n_lags = 20
    r = jnp.zeros([M, n_lags, B])
    r = r.at[:, 0, :].set(1)
    r = jax.lax.fori_loop(1, n_lags, step, r)

    # take a mean along chains and features
    r = r.mean(axis=[0, 2])
    
    return r


def r_hat(chains, f):
    """
    Based on the paper "What Are Bayesian Neural Network Posteriors Really Like? - Appendix B"
    chains: [num_chains x num_steps x num_params]
    f: f(chain_node) -> [B]: transforms params to a 1D quantity of interest, eg predictive distribution
    """
    
    # compute function of interest for each chain node
    # - y.shape: [num_chains (M) x num_steps (N) x num_features (B)]
    y = jnp.array([[f(params) for params in chain] for chain in chains])
    M, N, B = y.shape

    # compute variances
    b = (N/(M-1)) * ((y.mean(axis=1, keepdims=True) - y.mean(axis=[0, 1], keepdims=True))**2).sum(axis=[0, 1])
    w = 1/(M*(N-1)) * ((y - y.mean(axis=1, keepdims=True))**2).sum(axis=[0, 1])
    r_hat = jnp.sqrt((((N-1)/N)*w + (1/N)*b)/w)
    
    return r_hat