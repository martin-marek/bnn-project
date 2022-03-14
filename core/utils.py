import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def normal_like_tree(a, key):
    treedef = jax.tree_structure(a)
    num_vars = len(jax.tree_leaves(a))
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_multimap(lambda p, k: jax.random.normal(k, shape=p.shape), a, jax.tree_unflatten(treedef, all_keys))
    return noise


def ravel_pytree_(pytree):
    """Ravels a pytree like `jax.flatten_util.ravel_pytree`
    but doesn't return a function for unraveling."""
    leaves, treedef = tree_flatten(pytree)
    flat = jnp.concatenate([jnp.ravel(x) for x in leaves])
    return flat


def pmap_(f, shard_args, distr_args):
    # pmaps a function over `shard_args`, reusing `distr_args`
    def g(*shard_args):
        return f(*shard_args, *distr_args)
    return jax.pmap(g)(*shard_args)