import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "./MovieRecommender.css";
import Box from "@mui/material/Box";

import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import { List, Typography } from "@mui/material";
import { Container } from "@mui/material";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import InputLabel from "@mui/material/InputLabel";
import Button from "@mui/material/Button";

const predict = async (model, input) => {
  console.log("input", input);
  const n_occupations_in_movie_lens_dataset = 21;
  const n_zip_codes_in_movie_lens_dataset = 3439; 
  const max_gender_value = 1; // 0 or 1
  const max_age = 100; // 1 to 100


  const raw_inputs = tf.tensor(input);
  const features = raw_inputs.div(tf.tensor([max_age, max_gender_value, n_occupations_in_movie_lens_dataset, n_zip_codes_in_movie_lens_dataset]));
  const MovieQvalues = model.predict(features);
  const recommendedMovie = tf.argMax(MovieQvalues, 1).dataSync();
  console.log("MovieQvalues", MovieQvalues);
  console.log("recommendedMovie", recommendedMovie);
  return [recommendedMovie, MovieQvalues];
};

const setOptions = (path, setter) => {
  fetch(path)
    .then((res) => res.json())
    .then((data) => {
      const options = Object.keys(data).map((key) => ({
        value: data[key],
        label: key,
      }));
      setter(options);
    })
    .catch((err) => {
      console.log(err);
    });
};

function MovieRecommender() {
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState("Loading model...");

  const [movies, setMovies] = useState({});

  const [age, setAge] = useState("28");
  const [gender, setGender] = useState("F");
  const [occupation, setOccupation] = useState("");
  const [zipcode, setZipCode] = useState("");

  const [occupationOptions, setOccupationOptions] = useState({});
  const [genderOptions, setGenderOptions] = useState({});
  const [zipCodeOptions, setZipcodeOptions] = useState({});

  // load the occupations JSON file
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  const [top5MoviesWithQValues, setTop5MoviesWithQValues] = useState({});

  const computeTop5MoviesWithQValues = (MovieQvalues) => {
    const top5Movies = Object.keys(MovieQvalues)
      .sort((a, b) => MovieQvalues[b] - MovieQvalues[a])
      .slice(0, 5);
    const top5MoviesWithQValues = top5Movies.map((movie) => ({
      movie: movies[movie],
      QValue: MovieQvalues[movie],
    }));
    setTop5MoviesWithQValues(top5MoviesWithQValues);
  };

  const handleSubmit = async (e, age, gender, occupation, zipcode) => {
    e.preventDefault();
    const result = await predict(model, [[age, gender, occupation, zipcode]]);
    const values = result[1].dataSync();
    // const arr = Array.from(values);
    const obj = {};

    // Iterate over the values and format them with two decimal places
    for (let i = 0; i < values.length; i++) {
      obj[i] = values[i].toFixed(2);
    }
    computeTop5MoviesWithQValues(obj);
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

  useEffect(() => {
    const filenames_and_setters = [
      ["/assets/tfjs_model/occupations_to_idx.json", setOccupationOptions],
      ["/assets/tfjs_model/gender_to_idx.json", setGenderOptions],
      ["/assets/tfjs_model/zip_codes_to_idx.json", setZipcodeOptions],
    ];
    filenames_and_setters.forEach(([path, setter]) => setOptions(path, setter));
    setOptionsLoaded(true);
  }, []);
  if (!optionsLoaded) {
    return <div>Loading...</div>;
  }

  return (
    <Box flex={1}>
      <div
        style={{
          display: "flex",
          background: "linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)",
          height: "4rem",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <Box>
          <Typography
            style={{
              padding: "0",
              margin: "0",
              color: "white",
              fontSize: "1.5rem",
              fontWeight: "bold",
            }}
          >
            StreamRec - Movie Recommender System.
          </Typography>
          <Typography
            style={{
              padding: "0",
              margin: "0",
              color: "white",
            }}
            variant="caption"
          >
            {" "}
            Model Health: {status}
          </Typography>
        </Box>
      </div>
      <Box
        style={{
          display: "flex",
          justifyContent: "center",
          width: "100%",
          height: "100%",
        }}
      >
        <Box
          style={{
            width: "50%",
            height: "100%",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <Container maxWidth="sm">
            <Box
              component="form"
              sx={{
                "& > :not(style)": { m: 1, width: "25ch" },
              }} // 25ch = 400px
              noValidate
              autoComplete="off"
            >
              <Typography
                style={{
                  color: "black",
                  fontSize: "1.5rem",
                  fontWeight: "bold",
                  marginLeft: "1rem",
                }}
              >
                User Information
              </Typography>
              <InputLabel
                style={{
                  width: "100%",
                  marginTop: "1rem",
                  marginBottom: "0.5rem",
                  fontSize: "1rem",
                  fontWeight: "bold",
                  color: "black",
                  textAlign: "left",
                }}
                id="demo-simple-select-disabled-label"
              >
                Age
              </InputLabel>
              <TextField
                style={{ width: "100%" }}
                id="outlined-basic"
                label="Age"
                variant="outlined"
                type="number"
                inputProps={{ min: 1, max: 100, step: 1 }}
                onChange={(e) => setAge(parseFloat(e.target.value))}
              />
              <InputLabel
                style={{
                  width: "100%",
                  marginTop: "1rem",
                  marginBottom: "0.5rem",
                  fontSize: "1rem",
                  fontWeight: "bold",
                  color: "black",
                  textAlign: "left",
                }}
                id="demo-simple-select-disabled-label"
              >
                Gender
              </InputLabel>
              <Select
                style={{ width: "100%" }}
                label="Gender"
                value={gender}
                defaultValue={"F"}
                onChange={(e) => setGender(parseFloat(e.target.value))}
              >
                {Object.keys(genderOptions).map((key) => (
                  <MenuItem key={key} value={key}>
                    {genderOptions ? genderOptions[key].label : key}
                  </MenuItem>
                ))}
              </Select>
              <InputLabel
                style={{
                  width: "100%",
                  marginTop: "1rem",
                  marginBottom: "0.5rem",
                  fontSize: "1rem",
                  fontWeight: "bold",
                  color: "black",
                  textAlign: "left",
                }}
                id="demo-simple-select-disabled-label"
              >
                Occupation
              </InputLabel>
              <Select
                style={{ width: "100%" }}
                label="Occupation"
                value={occupation}
                onChange={(e) => setOccupation(parseFloat(e.target.value))}
              >
                {Object.keys(occupationOptions).map((key) => (
                  <MenuItem key={key} value={key}>
                    {occupationOptions ? occupationOptions[key].label : key}
                  </MenuItem>
                ))}
              </Select>

              <InputLabel
                style={{
                  width: "100%",
                  marginTop: "1rem",
                  marginBottom: "0.5rem",
                  fontSize: "1rem",
                  fontWeight: "bold",
                  color: "black",
                  textAlign: "left",
                }}
                id="demo-simple-select-disabled-label"
              >
                Zip Code
              </InputLabel>
              <Select
                labelId="demo-simple-select-disabled-label"
                style={{ width: "100%" }}
                label="Zipcode"
                onChange={(e) => setZipCode(parseFloat(e.target.value))}
                value={zipcode}
              >
                {Object.keys(zipCodeOptions).map((key) => (
                  <MenuItem key={key} value={key}>
                    {zipCodeOptions ? zipCodeOptions[key].label : key}
                  </MenuItem>
                ))}
              </Select>
              <Button
                variant="contained"
                onClick={(e) =>
                  handleSubmit(e, age, gender, occupation, zipcode)
                }
              >
                Submit
              </Button>
            </Box>
          </Container>
        </Box>
        <Box
          style={{
            width: "50%",
            height: "100%",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <Container maxWidth="sm">
            <Box
              component="form"
              sx={{
                "& > :not(style)": { m: 1 },
              }} // 25ch = 400px
              noValidate
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                justifyContent: "center",
                alignItems: "center",
              }}
              autoComplete="off"
            >
              <Typography
                style={{
                  color: "black",
                  fontSize: "1.5rem",
                  fontWeight: "bold",
                  marginLeft: "1rem",
                }}
              >
                Movie Recommendations
              </Typography>
              <Box
                style={{
                  width: "100%",
                  height: "100%",
                  display: "flex",
                  flexDirection: "row",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <Box
                  style={{
                    width: "50%",
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    alignItems: "center",
                  }}
                >
                  <Typography
                    style={{
                      color: "black",
                      fontSize: "1rem",
                      fontWeight: "bold",
                      marginLeft: "1rem",
                    }}
                  >
                    Top 5 Movies
                  </Typography>
                  <Box
                    style={{
                      width: "100%",
                      height: "100%",
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "center",
                      alignItems: "center",
                    }}
                  >
                    <List>
                      {top5MoviesWithQValues &&
                        // Map over object with key and values and display top 5 movies
                        // display as list
                        Object.keys(top5MoviesWithQValues).map((movie) => (
                          <ListItem key={movie}>
                            <ListItemText
                              primary={top5MoviesWithQValues[movie].movie}
                              secondary={
                                "QValue: " + top5MoviesWithQValues[movie].QValue
                              }
                            />
                          </ListItem>
                        ))}
                    </List>
                  </Box>
                </Box>
              </Box>
            </Box>
          </Container>
        </Box>
      </Box>
    </Box>
  );
}

export default MovieRecommender;
