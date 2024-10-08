import React from "react";
import ReactDOM from "react-dom";
import MyComponent from "./MyComponent";

ReactDOM.render(
  <React.StrictMode>
    <h1>Hello world</h1>
    <MyComponent />
  </React.StrictMode>,
  document.getElementById("root")
);