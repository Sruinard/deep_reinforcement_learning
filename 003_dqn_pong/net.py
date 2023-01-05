import flax.linen as nn
import config


class Brain(nn.Module):

    hparams: config.HParams

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=4,
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=2,
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=1,
                    kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.leaky_relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hparams.hidden_layer_n_neurons)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(features=self.hparams.n_actions)(x)
        return x
