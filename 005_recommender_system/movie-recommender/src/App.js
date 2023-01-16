import MovieRecommender from "./MovieRecommender";
// import UserInput from "./UserInput";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        {
          <>
            <MovieRecommender />
            {/* <UserInput /> */}
          </>
        }
      </header>
    </div>
  );
}

export default App;
