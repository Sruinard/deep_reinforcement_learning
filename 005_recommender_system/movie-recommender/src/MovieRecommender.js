import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "./MovieRecommender.css";
import UserInput from "./UserInput";

const predict = async (model, input) => {
  console.log("input", input);
  const features = tf.tensor(input);
  const MovieQvalues = model.predict(features);
  const recommendedMovie = tf.argMax(MovieQvalues, 1).dataSync();
  console.log("MovieQvalues", MovieQvalues);
  console.log("recommendedMovie", recommendedMovie);
  return recommendedMovie;
};

function MovieRecommender() {
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState("Loading model...");

  const [movies, setMovies] = useState({});
  const [recommendedMovie, setRecommendedMovie] = useState(null);

  const handleSubmit = async (e, age, gender, occupation, zipcode) => {
    e.preventDefault();
    const result = await predict(model, [[age, gender, occupation, zipcode]]);
    setRecommendedMovie(result);
  };

  useEffect(() => {
    async function loadModel() {
      const model = await tf.loadGraphModel("/assets/tfjs_model/model.json", {
        onProgress: (p) => setStatus(`loading model: ${Math.round(p * 100)}%`),
      });
      setModel(model);
    }
    loadModel();
  }, []);

  useEffect(() => {
    async function loadMovies() {
      const response = await fetch("/assets/movie_id_to_title.json");
      const data = await response.json();
      setMovies(data);
    }
    loadMovies();
  }, []);

  const a_movie = movies[2];

  return (
    <div>
      <h1>Movie Recommender</h1>
      <p>Model: {model ? "Loaded" : status}</p>
      <UserInput handleSubmit={handleSubmit} />
      <h1>Recommended Movie: {recommendedMovie}</h1>
      <p>A Movie: {a_movie}</p>
      {recommendedMovie && (
        <div>
          <h1>Recommended Movie Title: {recommendedMovie}</h1>
          <h2>{movies[recommendedMovie[0] + 1]}</h2>
        </div>
      )}
    </div>
  );
}

export default MovieRecommender;
