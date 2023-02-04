import pytest

import ppo_main

@pytest.fixture
def buffer():
    buffer = ppo_main.Buffer.create(
        horizon=64,
        observation_shape=(4,)
    )
    return buffer

def test_train_agent(buffer):

    train_config = ppo_main.TrainConfig()
    state = ppo_main.create_train_state(
        train_config=train_config,
        observation_shape=(4,),
        num_actions=2,
    )

    out = ppo_main.update_ppo_model(
        state=state,
        buffer=buffer,
        train_config=train_config
    )


