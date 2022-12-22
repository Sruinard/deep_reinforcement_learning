import gymnasium as gym
import flax.linen as nn
from flax.training import checkpoints, train_state
# from flax.metrics import tensorboard
from torch.utils import tensorboard
import optax
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax
from dataclasses import dataclass, field
from typing import List
from tqdm.auto import tqdm

# Create main components of RL


@dataclass
class EpisodeStep:
    observation: jax.Array
    action: int


@dataclass
class Episode:
    reward: int = 0
    steps: List[EpisodeStep] = field(default_factory=list)


class NNAgent(nn.Module):
    n_hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        # used for training
        logits = nn.Dense(self.n_actions)(x)
        # used for taking actions
        action_probs = nn.softmax(logits, axis=1)
        return {
            "logits": logits,
            "action_probs": action_probs
        }

# Dataset based on the agent interacting with the environment
# Note: it's important to have a separate rng at every step
# otherwise actions will be correlated and all actions will move to
# 0 or 1


class InteractionDataset:

    def __init__(self, batch_size, env, percentile, seed=42):
        self.batch_size = batch_size
        self.env = env
        self.percentile = percentile
        self.seed = seed

    def sample_action(self, state, observation, action_rng):
        obs = jnp.asarray(observation)[None, :]
        action_probs = state.apply_fn({"params": state.params}, obs)[
            'action_probs'][0]
        action = jrandom.choice(
            action_rng, a=len(action_probs), p=action_probs).item()
        return action

    def generate_episode(self, state, rng):
        episode = Episode()
        observation = self.env.reset()[0]
        while True:
            rng, action_rng = jrandom.split(rng)
            action = self.sample_action(state, observation, action_rng)
            next_observation, reward, is_done, _, _ = self.env.step(action)
            episode.reward += reward
            episode.steps.append(EpisodeStep(
                observation=observation, action=action))
            if is_done:
                return episode
            observation = next_observation

    def filter_by_percentile(self, batch):
        rewards = jnp.array([episode.reward for episode in batch])
        is_positive_bound = jnp.percentile(rewards, self.percentile)
        features, labels = [], []
        for episode in batch:
            if episode.reward < is_positive_bound:
                continue
            features.extend([step.observation for step in episode.steps])
            labels.extend([step.action for step in episode.steps])
        return jnp.asarray(features), jnp.asarray(labels), rewards.mean(), is_positive_bound

    def get_batch(self, state, rng):
        batch = []
        while len(batch) != self.batch_size:
            rng, episode_rng = jrandom.split(rng)
            episode = self.generate_episode(state, episode_rng)
            batch.append(episode)
        features, labels, reward_mean, is_positive_bound = self.filter_by_percentile(
            batch)
        return features, labels, reward_mean, is_positive_bound

# Trainer


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 0.01
    percentile: int = 70
    n_steps: int = 100


class Trainer:

    def __init__(self, model_class, model_hparams, train_config: TrainConfig, log_dir: str = "./assets"):
        self.model_hparams = model_hparams
        self.model = model_class(**model_hparams)
        self.seed = 42
        self.summary_writer = tensorboard.SummaryWriter(log_dir=log_dir)
        self.train_config = train_config
        self.create_functions()

    def init_model(self, batch_example):
        init_rng = jrandom.PRNGKey(self.seed)
        variables = self.model.init(init_rng, batch_example)
        return variables

    def init_optimizer(self):
        return optax.adam(learning_rate=self.train_config.lr)

    def create_train_state(self, batch_example):
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_model(batch_example=batch_example)["params"],
            tx=self.init_optimizer()
        )
        return state

    def create_functions(self):

        @jax.jit
        def apply_model(state, features, labels):
            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    features
                )['logits']
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits, labels).mean()
                acc = (logits.argmax(axis=-1) == labels).mean()
                return loss, acc
            (loss, acc), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            return grads, loss, acc

        def train_step(state, features, labels):
            grads, loss, acc = apply_model(state, features, labels)
            return state.apply_gradients(grads=grads), loss, acc

        def eval_step(state, features, labels):
            _, loss, acc = apply_model(state, features, labels)
            return loss, acc
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def train_and_eval(self, state, dataloader: InteractionDataset, rng):

        for step_idx in tqdm(range(self.train_config.n_steps)):
            rng, batch_rng = jrandom.split(rng)
            features, labels, reward_mean, reward_bound = dataloader.get_batch(
                state=state, rng=batch_rng)
            state, loss, acc = self.train_step(state, features, labels)

            self.summary_writer.add_scalar(
                "reward", np.array(reward_mean), step_idx)
            self.summary_writer.add_scalar(
                "reward_bound", np.array(reward_bound), step_idx)
            self.summary_writer.add_scalar("loss", np.array(loss), step_idx)
            self.summary_writer.add_scalar(
                "action_mean", np.array(labels.mean()), step_idx)

            if reward_mean > 199:
                print("Solved Environment!")
                break

            print(
                f"""\nTraining: 
                \n\t Mean Reward {reward_mean} 
                \n\t 70 Percentile: {reward_bound}
                \n\t loss: {loss}
                \n\t acc: {acc}
                \n\t action_mean: {labels.mean()}
                """)


if __name__ == "__main__":
    # create environment
    environment = gym.make("CartPole-v1")

    # training hyperparameters
    train_config = TrainConfig()
    dataloader = InteractionDataset(
        batch_size=train_config.batch_size, env=environment, percentile=train_config.percentile, seed=42)

    # model hyperparameters
    model_hparams = {
        "n_hidden": 128,
        "n_actions": environment.action_space.n
    }
    trainer = Trainer(model_class=NNAgent,
                      model_hparams=model_hparams, train_config=train_config)

    # initialize model
    rng = jrandom.PRNGKey(42)
    rng, batch_rng = jrandom.split(rng)
    batch_example = jrandom.normal(
        batch_rng, (train_config.batch_size, environment.observation_space.shape[0]))
    state = trainer.create_train_state(batch_example)

    # train and evaluate
    trainer.train_and_eval(state=state, dataloader=dataloader, rng=rng)
