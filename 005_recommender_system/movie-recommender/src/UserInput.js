import React, { useEffect, useState } from "react";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";

// const genderOptions = {
//   Male: "male",
//   Female: "female",
//   Other: "other",
// };

// const occupationOptions = {
//   Student: "student",
//   Professional: "professional",
//   Retired: "retired",
// };

// const zipCodeOptions = {
//   12345: "12345",
//   54321: "54321",
//   67890: "67890",
// };

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

const UserInput = (props) => {
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [occupation, setOccupation] = useState("");
  const [zipcode, setZipCode] = useState("");

  const [occupationOptions, setOccupationOptions] = useState({});
  const [genderOptions, setGenderOptions] = useState({});
  const [zipCodeOptions, setZipcodeOptions] = useState({});

  // load the occupations JSON file
  const [optionsLoaded, setOptionsLoaded] = useState(false);
  useEffect(() => {
    const filenames_and_setters = [
      ["/assets/tfjs_model/occupations_to_idx.json", setOccupationOptions],
      ["/assets/tfjs_model/gender_to_idx.json", setGenderOptions],
      ["/assets/tfjs_model/zipcodes_to_idx.json", setZipcodeOptions],
    ];
    filenames_and_setters.forEach(([path, setter]) => setOptions(path, setter));
    setOptionsLoaded(true);
  }, []);
  if (!optionsLoaded) {
    return <div>Loading...</div>;
  }

  return (
    <>
      <form
        onSubmit={(e) =>
          props.handleSubmit(
            e,
            parseFloat(age),
            parseFloat(gender),
            parseFloat(occupation),
            parseFloat(zipcode)
          )
        }
      >
        <h1>Movie Recommender</h1>
        <TextField
          label="Age"
          type="number"
          value={age}
          onChange={(e) => setAge(e.target.value)}
        />

        <Select value={gender} onChange={(e) => setGender(e.target.value)}>
          {Object.keys(genderOptions).map((key) => (
            <MenuItem key={key} value={genderOptions[key]}>
              {genderOptions ? genderOptions[key].label : key}
            </MenuItem>
          ))}
        </Select>
        <Select
          value={occupation}
          onChange={(e) => setOccupation(e.target.value)}
        >
          {Object.keys(occupationOptions).map((key) => (
            <MenuItem key={key} value={occupationOptions[key]}>
              {occupationOptions ? occupationOptions[key].label : key}
            </MenuItem>
          ))}
        </Select>

        <Select value={zipcode} onChange={(e) => setZipCode(e.target.value)}>
          {Object.keys(zipCodeOptions).map((key) => (
            <MenuItem key={key} value={zipCodeOptions[key]}>
              {zipCodeOptions ? zipCodeOptions[key].label : key}
            </MenuItem>
          ))}
        </Select>
        <button type="submit">Submit</button>
      </form>
    </>
  );
};

export default UserInput;
