import flax.linen as nn


class ActorCritic(nn.Module):

    n_actions: int

    def setup(self):
        self.backbone = nn.Sequential([
            nn.Dense(features=128, kernel_init=nn.initializers.kaiming_normal()),
            nn.leaky_relu
        ]
        )

        self.policy_head = nn.Sequential(
            [
                nn.Dense(features=self.n_actions)
            ]
        )

        self.value_head = nn.Sequential([
            nn.Dense(features=1)
        ]
        )

    def __call__(self, x):
        x = self.backbone(x)
        x = x.reshape((x.shape[0], -1))
        policy_logits = self.policy_head(x)
        state_values = self.value_head(x)
        return policy_logits, state_values
