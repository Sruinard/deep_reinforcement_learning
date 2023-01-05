import pytest
import functools
import jax
import jax.random as jrandom
import jax.numpy as jnp
import environment
import agent
import gymnasium as gym


@pytest.fixture
def batch():
    rng = jrandom.PRNGKey(42)
    rng, state_rng, actions_rng, rewards_rng, is_dones_rng, next_states_rng = jrandom.split(
        rng, 6)
    batch_size = 16

    return (
        jrandom.normal(state_rng, (batch_size, 4, 84, 84, 1)),
        jrandom.choice(actions_rng, jnp.arange(6), shape=(batch_size,)),
        jrandom.choice(rewards_rng, jnp.arange(2), shape=(batch_size,)),
        jrandom.choice(is_dones_rng, jnp.asarray(
            [True, False]), shape=(batch_size,)),
        jrandom.normal(next_states_rng, (batch_size, 4, 84, 84, 1)),
    )


@pytest.fixture
def interactor(batch):
    model = agent.Agent(
        model_architecture=agent.Brain,
        model_hparams=agent.ModelHParams(),
        agent_hparams=agent.AgentHParams(),
        env=environment.create("PongNoFrameskip-v4"),
        n_actions=6
    )
    state = model.initialize(batch[0])
    target_state = model.initialize(batch[4])
    return model, state, target_state


def test_batch_size(batch):
    assert batch[0].shape == (16, 4, 84, 84, 1)
    assert batch[1].shape == (16,)
    assert batch[2].shape == (16,)
    assert batch[3].shape == (16,)
    assert batch[4].shape == (16, 4, 84, 84, 1)


def test_train_step(batch, interactor):
    model, state, target_state = interactor
    loss, state = agent.train_step(
        batch=batch,
        state=state,
        target_state=target_state,
        gamma=0.99
    )
    assert loss is not None
    assert state.params is not None


def test_can_sync_target_network(interactor):
    model, state, target_state = interactor
    target_state = agent.sync_target_network(state, target_state)
    assert target_state.params == state.params


def test_can_save_and_load_model(interactor):
    model, state, target_state = interactor
    agent.save_model(state, "./checkpoints/test_model")
    loaded_params = agent.load_model("./checkpoints/test_model")
    target_state.replace(params=loaded_params)
    forward_pass_before_saving = state.apply_fn({"params": state.params}, jnp.ones((1, 4, 84, 84, 1)))
    forward_pass_after_loading = target_state.apply_fn({"params": target_state.params}, jnp.ones((1, 4, 84, 84, 1)))
    assert (forward_pass_before_saving == forward_pass_after_loading).all()
