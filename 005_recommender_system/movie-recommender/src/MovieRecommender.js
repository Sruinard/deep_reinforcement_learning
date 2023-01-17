import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "./MovieRecommender.css";
import UserInput from "./UserInput";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";

import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import { FixedSizeList } from "react-window";

// function renderRow(props) {
//   const { index, style, MovieQvalues, movies } = props;
//   console.log("MovieQvalues", MovieQvalues);
//   // const valueAtIndex5 = MovieQValues[5];

//   return (
//     <ListItem style={style} key={index} component="div" disablePadding>
//       <ListItemButton>
//         <ListItemText
//           primary={`Title: ${movies[index + 1]} - QValue: ${
//             MovieQvalues[index]
//           }`}
//         />
//       </ListItemButton>
//     </ListItem>
//   );
// }

// function VirtualizedList(props) {
//   const { MovieQvalues, movies } = props;
//   const n_movies = Object.keys(movies).length;
//   return (
//     <Box
//       sx={{
//         width: "100%",
//         height: 400,
//         maxWidth: 360,
//         bgcolor: "background.paper",
//       }}
//     >
//       <FixedSizeList
//         height={400}
//         width={360}
//         itemSize={46}
//         itemCount={n_movies}
//         overscanCount={5}
//       >
//         {(renderProps) => renderRow({ ...renderProps, MovieQvalues, movies })}
//       </FixedSizeList>
//     </Box>
//   );
// }

function renderRow(props) {
  const { index, style, moviesAndQValues } = props;
  const { movie, QValue } = moviesAndQValues[index];

  return (
    <ListItem style={style} key={index} component="div" disablePadding>
      <ListItemButton>
        <ListItemText primary={`Title: ${movie} - QValue: ${QValue}`} />
      </ListItemButton>
    </ListItem>
  );
}

function VirtualizedList(props) {
  const { MovieQvalues, movies } = props;

  // Create an array of objects that contains the movie title, its corresponding Q-value, and its original index
  const moviesAndQValues = Object.keys(movies).map((key, index) => ({
    movie: movies[key],
    QValue: MovieQvalues[index],
    index,
  }));

  // Sort the array of objects based on the Q-value in descending order
  moviesAndQValues.sort((a, b) => b.QValue - a.QValue);

  return (
    <Box
      sx={{
        width: "100%",
        height: 400,
        maxWidth: 360,
        bgcolor: "background.paper",
      }}
    >
      <FixedSizeList
        height={400}
        width={360}
        itemSize={46}
        itemCount={moviesAndQValues.length}
        overscanCount={5}
      >
        {(renderProps) =>
          renderRow({
            ...renderProps,
            moviesAndQValues,
          })
        }
      </FixedSizeList>
    </Box>
  );
}

const predict = async (model, input) => {
  console.log("input", input);
  const features = tf.tensor(input);
  const MovieQvalues = model.predict(features);
  const recommendedMovie = tf.argMax(MovieQvalues, 1).dataSync();
  console.log("MovieQvalues", MovieQvalues);
  console.log("recommendedMovie", recommendedMovie);
  return [recommendedMovie, MovieQvalues];
};

function MovieRecommender() {
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState("Loading model...");

  const [movies, setMovies] = useState({});
  const [recommendedMovie, setRecommendedMovie] = useState(null);
  const [MovieQvalues, setMovieQvalues] = useState(null);

  const handleSubmit = async (e, age, gender, occupation, zipcode) => {
    e.preventDefault();
    const result = await predict(model, [[age, gender, occupation, zipcode]]);
    setRecommendedMovie(result[0]);
    const values = result[1].dataSync();
    // const arr = Array.from(values);
    const obj = {};

    // Iterate over the values and format them with two decimal places
    for (let i = 0; i < values.length; i++) {
      obj[i] = values[i].toFixed(2);
    }
    setMovieQvalues(obj);
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
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <UserInput handleSubmit={handleSubmit} />
          {recommendedMovie && (
            <div>
              <h1>Movies to pick from</h1>
              <VirtualizedList MovieQvalues={MovieQvalues} movies={movies} />
            </div>
          )}
        </div>
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
            <h1>Top Recommended Movie</h1>
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
