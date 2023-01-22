import os
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import json
from typing import Dict

N_USERS = 943
N_MOVIES = 1682

# gym class with each state representing a user from the MovieLens dataset


class MovieLensEnv(gym.Env):
    def __init__(self, data_dir: str, n_actions: int, train: bool = True):
        # age, gender, occupation, zip code
        self.data_dir = data_dir
        self.n_actions = n_actions
        self.train = train
        self.user_movies_ratings = self._load_user_movie_ratings_matrix()[
            :, :n_actions]
        self.user_observations = self._load_user_features_from_file(
        ) / jnp.asarray([100, 1, self.n_occupations, self.n_zip_codes])
        self.movies = self._load_movies_from_file()
        self.prev_user_id = None
        self.user_id = None
        self.n_steps = 0
        self.rng = jax.random.PRNGKey(0)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([100, 1, self.n_occupations, self.n_zip_codes]))
        self.action_space = gym.spaces.Discrete(n_actions)
        self.save_files_for_inference()
        self.reset()

    def _load_user_movie_ratings_matrix(self, filename: str = 'u1.base'):
        user_movie_ratings_matrix = np.zeros((N_USERS, N_MOVIES))
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            for line in f:
                user_id, movie_id, rating, _ = line.split('\t')

                user_id = int(user_id) - 1  # user ids start from 1
                movie_id = int(movie_id) - 1  # movie ids start from 1
                rating = float(rating)
                user_movie_ratings_matrix[user_id, movie_id] = rating

        return jnp.asarray(user_movie_ratings_matrix)

    def _load_user_features_from_file(self, filename: str = 'u.user'):
        """Load users features from the movie lens dataset.
        file has the following format:
         user id | age | gender | occupation | zip code
        """
        zip_codes = {}
        occupations = {}
        zip_code_id = 0
        occupation_id = 0

        n_features = 4
        user_observations = np.zeros((N_USERS, n_features))
        self.gender_to_idx = {"M": 0, "F": 1}

        with open(os.path.join(self.data_dir, filename), 'r') as f:
            for line in f:
                user_id, age, gender, occupation, zip_code = line.split('|')
                user_id = int(user_id) - 1  # user ids start from 1
                age = int(age)
                zip_code = zip_code.strip()

                user_observations[user_id, 0] = age
                user_observations[user_id, 1] = self.gender_to_idx[gender]
                user_observations[user_id, 2] = occupation_id
                user_observations[user_id, 3] = zip_code_id

                # needed for inference
                occupations[occupation] = occupation_id
                occupation_id += 1
                zip_codes[zip_code] = zip_code_id
                zip_code_id += 1

        zip_codes["other"] = zip_code_id
        zip_code_id += 1

        self.n_occupations = occupation_id
        self.n_zip_codes = zip_code_id
        self.occupations_to_idx = occupations
        self.zip_codes_to_idx = zip_codes

        return jnp.asarray(user_observations)

    def _load_movies_from_file(self, filename: str = 'u.item'):
        """Load movies from the movie lens dataset.
        file has the following format:
            movie id | movie title | release date | video release date |
            IMDb URL | unknown | Action | Adventure | Animation |
            Children's | Comedy | Crime | Documentary | Drama | Fantasy |
            Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
            """
        movies = {}
        with open(os.path.join(self.data_dir, filename), 'r', encoding="ISO-8859-1") as f:
            for line in f:
                movie_id, title, release_date, video_release_date, \
                    imdb_url, *genres = line.split('|')
                movie_id = int(movie_id) - 1  # movie ids start from 1
                if movie_id not in movies:
                    movies[movie_id] = {}
                movies[movie_id]['title'] = title
                movies[movie_id]['release_date'] = release_date
                movies[movie_id]['video_release_date'] = video_release_date
                movies[movie_id]['imdb_url'] = imdb_url
                movies[movie_id]['genres'] = genres

        return movies

    def step(self, action: int):
        reward = self.regret(self.prev_user_id, action)
        user_id = np.random.choice(self.user_movies_ratings.shape[0])
        observation = self.user_observations[user_id]
        self.prev_user_id = user_id

        self.n_steps += 1
        done = self.n_steps == self.user_movies_ratings.shape[0]
        truncated = {}
        info = {}

        return observation, reward, done, truncated, info

    def reward(self, user_id: int, movie_id: int):
        return self.user_movies_ratings[user_id, movie_id]

    def regret(self, user_id: int, movie_id: int):
        max_user_reward = jnp.max(self.user_movies_ratings[user_id, :])
        return max_user_reward - self.reward(user_id, movie_id)

    def reset(self, seed=None, options=None):
        self.prev_user_id = np.random.choice(self.user_movies_ratings.shape[0])
        observation = self.user_observations[self.prev_user_id]
        info = {}
        self.n_steps = 0
        return observation, info

    def save_vocab_to_idx_mapping_as_json(self, vocab_to_idx: Dict[str, int], path: str):
        with open(path, 'w') as f:
            json.dump(vocab_to_idx, f)

    def save_files_for_inference(self):
        if not os.path.exists("./assets/tfjs_model"):
            os.makedirs("./assets/tfjs_model")
        self.save_vocab_to_idx_mapping_as_json(
            self.occupations_to_idx,
            path="./assets/tfjs_model/occupations_to_idx.json"
        )
        self.save_vocab_to_idx_mapping_as_json(
            self.zip_codes_to_idx,
            path="./assets/tfjs_model/zip_codes_to_idx.json"
        )
        self.save_vocab_to_idx_mapping_as_json(
            self.gender_to_idx, path="./assets/tfjs_model/gender_to_idx.json"
        )
        self.save_vocab_to_idx_mapping_as_json({movie_id: movie_attributes["title"] for movie_id, movie_attributes in self.movies.items()},
                                               path="./assets/movie_id_to_title.json"
                                               )
