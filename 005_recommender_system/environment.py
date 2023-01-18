import os
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import json
from typing import Dict


# gym class with each state representing a user from the MovieLens dataset
class MovieLensEnv(gym.Env):
    def __init__(self, data_dir: str, n_actions: int, train: bool = True):
        # age, gender, occupation, zip code
        self.data_dir = data_dir
        self.n_actions = n_actions
        self.train = train
        self.user_ratings = self._load_user_ratings_from_file()
        self.user_features, self.n_occupations, self.n_zip_codes, self.occupations_to_idx, self.zip_codes_to_idx = self._load_user_features_from_file()
        self.movies = self._load_movies_from_file()
        self.users_ids = list(self.user_ratings.keys())
        self.movies_ids = list(self.movies.keys())
        self.prev_user_id = None
        self.user_id = None
        self.n_steps = 0
        self.rng = jax.random.PRNGKey(0)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([100, 1, self.n_occupations, self.n_zip_codes]), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_actions)
        self.save_files_for_inference()
        self.reset()

    def _load_user_ratings_from_file(self, filename: str = 'u1.base'):
        user_ratings = {}
        n_users = 0
        with open(os.path.join(self.data_dir, filename), 'r') as f:
            for line in f:
                user_id, movie_id, rating, _ = line.split('\t')

                user_id = int(user_id)
                movie_id = int(movie_id)
                rating = float(rating)
                # make action space smaller to simplify the problem scope
                if movie_id > self.n_actions:
                    continue
                if user_id not in user_ratings:
                    user_ratings[user_id] = {}
                user_ratings[user_id][movie_id] = rating
                n_users += 1

        return user_ratings

    def _load_user_features_from_file(self, filename: str = 'u.user'):
        """Load users features from the movie lens dataset.
        file has the following format:
         user id | age | gender | occupation | zip code
        """
        user_features = {}
        zip_codes = {}
        occupations = {}
        zip_code_id = 0
        occupation_id = 0

        with open(os.path.join(self.data_dir, filename), 'r') as f:
            for line in f:
                user_id, age, gender, occupation, zip_code = line.split('|')
                user_id = int(user_id)
                age = int(age)
                zip_code = zip_code.strip()
                if user_id not in user_features:
                    user_features[user_id] = {}
                user_features[user_id]['age'] = age
                user_features[user_id]['gender'] = gender
                user_features[user_id]['occupation'] = occupation
                user_features[user_id]['zip_code'] = zip_code

                occupations[occupation] = occupation_id
                occupation_id += 1
                zip_codes[zip_code] = zip_code_id
                zip_code_id += 1

        zip_codes["other"] = zip_code_id
        zip_code_id += 1

        return user_features, occupation_id, zip_code_id, occupations, zip_codes

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
                movie_id = int(movie_id)
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
        user_id = np.random.choice(np.asarray(self.users_ids))
        observation = self.user_features[user_id]
        self.prev_user_id = user_id

        self.n_steps += 1
        done = self.n_steps == len(self.user_ratings)
        truncated = {}
        info = {}

        return self.obs_to_jnp_array(observation), reward, done, truncated, info

    def reward(self, user_id: int, movie_id: int):
        return self.user_ratings[user_id].get(movie_id, 0.0)

    def regret(self, user_id: int, movie_id: int):
        max_user_reward = max(list(self.user_ratings[user_id].values()))
        return max_user_reward - self.reward(user_id, movie_id)

    def reset(self, seed=None, options=None):
        self.prev_user_id = np.random.choice(
            np.asarray(self.users_ids))
        observation = self.user_features[self.prev_user_id]
        info = {}
        self.n_steps = 0
        return self.obs_to_jnp_array(observation), info

    def gender_to_idx(self, gender):
        return {
            "M": 0,
            "F": 1
        }[gender]

    def obs_to_jnp_array(self, obs):
        return jnp.array([obs['age'], self.gender_to_idx(obs['gender']), self.occupations_to_idx.get(obs['occupation'], self.n_occupations + 1), self.zip_codes_to_idx.get(obs['zip_code'], self.n_zip_codes + 1)])

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
            {"M": 0, "F": 1}, path="./assets/tfjs_model/gender_to_idx.json"
        )
        self.save_vocab_to_idx_mapping_as_json({movie_id: movie_attributes["title"] for movie_id, movie_attributes in self.movies.items()},
                                               path="./assets/movie_id_to_title.json"
                                               )
