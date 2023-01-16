import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "./MovieRecommender.css";
import UserInput from "./UserInput";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
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

  return (
    <div>
      {/* div place title in center */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          flexDirection: "column",
          padding: "20px",
        }}
      >
        <h1 style={{ margin: "0px" }}>Movie Recommender</h1>
        <p>{model ? "Movie Recommendations using your RL-Agent" : status}</p>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          flexDirection: "column",
        }}
      >
        <UserInput handleSubmit={handleSubmit} />
      </div>
      {recommendedMovie && (
        <div>
          <Box
            sx={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              justifyContent: "center",
              "& > :not(style)": {
                m: 1,
                width: 256,
                height: 128,
              },
            }}
          >
            <Paper
              elevation={10}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {movies[recommendedMovie[0] + 1]}
            </Paper>
          </Box>
        </div>
      )}
    </div>
  );
}

export default MovieRecommender;
