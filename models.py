import jax
import haiku as hk


def make_mlp_fn(layer_dims, output_dim):
    """Returns a forward function for an MLP of given dimensions."""

    def forward(x):
        """
        Input: [B, C, L]
        Output: [B, output_dim]]
        """

        # flatten input
        x = hk.Flatten()(x) # [B, C, L] -> [B, C*L]

        # hidden layers
        for layer_dim in layer_dims:
            x = hk.Linear(layer_dim)(x) # [B, layer_dim]
            x = jax.nn.relu(x)

        # last layer: output corresponds to predicted mean and std 
        x = hk.Linear(output_dim)(x) # [B, 2]

        # apply softplus to std
        x = x.at[:, 1].set(jax.nn.softplus(x[:, 1]))

        return x

    return forward


def make_predict_fn(net):
    def predict_fn(x, params):
        y_hat, _ = net.apply(params, None, None, x)
        return y_hat
    return predict_fn


def make_nn(x, layer_dims=[100, 100, 100], output_dim=2):
    net_fn = make_mlp_fn(layer_dims, output_dim)
    net = hk.transform_with_state(net_fn)
    key, net_init_key = jax.random.split(jax.random.PRNGKey(0), 2)
    params, _ = net.init(net_init_key, x)
    predict_fn = make_predict_fn(net)
    return predict_fn, params
