import React, { useEffect, useState } from "react";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import InputLabel from "@mui/material/InputLabel";
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
  const [age, setAge] = useState("28");
  const [gender, setGender] = useState("F");
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
      ["/assets/tfjs_model/zip_codes_to_idx.json", setZipcodeOptions],
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
        <h1>100 Movies to recommend</h1>
        <h5
          style={{
            textAlign: "center",
          }}
        >
          For a small data exchange, we help you find the movies that best fit
          you!
        </h5>

        <InputLabel
          style={{
            alignSelf: "flex-end",
            textAlign: "end",
            color: "black",
          }}
          id="demo-simple-select-label"
        >
          Age
        </InputLabel>
        <TextField
          style={{ width: "100%" }}
          label="Age"
          type="number"
          value={age}
          onChange={(e) => setAge(e.target.value)}
        />

        <InputLabel
          style={{
            alignSelf: "flex-end",
            textAlign: "end",
            color: "black",
          }}
          id="demo-simple-select-label"
        >
          Gender
        </InputLabel>
        <Select
          style={{ width: "100%" }}
          label="Gender"
          value={gender}
          onChange={(e) => setGender(e.target.value)}
        >
          {Object.keys(genderOptions).map((key) => (
            <MenuItem key={key} value={genderOptions[key]}>
              {genderOptions ? genderOptions[key].label : key}
            </MenuItem>
          ))}
        </Select>

        <InputLabel
          style={{
            alignSelf: "flex-end",
            textAlign: "end",
            color: "black",
          }}
          id="demo-simple-select-label"
        >
          Occupation
        </InputLabel>

        <Select
          style={{ width: "100%" }}
          value={occupation}
          onChange={(e) => setOccupation(e.target.value)}
        >
          {Object.keys(occupationOptions).map((key) => (
            <MenuItem key={key} value={occupationOptions[key]}>
              {occupationOptions ? occupationOptions[key].label : key}
            </MenuItem>
          ))}
        </Select>
        <InputLabel
          style={{
            alignSelf: "flex-end",
            textAlign: "end",
            color: "black",
          }}
          id="demo-simple-select-label"
        >
          Zip Code
        </InputLabel>
        <Select
          style={{ width: "100%" }}
          value={zipcode}
          onChange={(e) => setZipCode(e.target.value)}
        >
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
