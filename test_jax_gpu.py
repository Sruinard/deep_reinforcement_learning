import jax
from jax.lib import xla_bridge

from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".8"

print(xla_bridge.get_backend().platform)


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(jax.random.PRNGKey(0), batch)

import time
startTime = time.time()

for _ in range(1000):
    output = model.apply(variables, batch)
print(output)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
