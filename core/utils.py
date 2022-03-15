import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten


def ifelse(cond, val_true, val_false):
    return jax.lax.cond(cond, lambda x: x[0], lambda x: x[1], (val_true, val_false))


def ravel_pytree_(pytree):
    """Ravels a pytree like `jax.flatten_util.ravel_pytree`
    but doesn't return a function for unraveling."""
    leaves, treedef = tree_flatten(pytree)
    flat = jnp.concatenate([jnp.ravel(x) for x in leaves])
    return flat
