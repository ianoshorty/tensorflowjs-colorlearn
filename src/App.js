import React, { Component } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";

class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      prediction: null,
      loss: null,
      loop: 0,
      has_red: 0,
      has_white: 0,
      has_yellow: 0
    };

    this.setupModel();
    this.trainModel(100);
  }

  setupModel() {
    // Define a model for linear regression.
    this.model = tf.sequential();
    this.model.add(
      tf.layers.dense({ units: 8, inputShape: [3], activation: "relu" })
    );
    this.model.add(tf.layers.dense({ units: 5, activation: "softmax" }));

    // Prepare the model for training: Specify the loss and the optimizer.
    this.model.compile({
      loss: "categoricalCrossentropy",
      optimizer: tf.train.adam(0.01)
    });
  }

  async trainModel(iterations = 1) {
    let pink = [1, 0, 0, 0, 0];
    let red = [0, 1, 0, 0, 0];
    let white = [0, 0, 1, 0, 0];
    let green = [0, 0, 0, 1, 0];
    let yellow = [0, 0, 0, 0, 1];

    // Generate some synthetic data for training.
    const inputs = tf.tensor2d([
      [1, 1, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
      [1, 0, 1]
    ]);
    const outputs = tf.tensor2d([
      green,
      red,
      white,
      yellow,
      yellow,
      pink,
      green
    ]);

    for (let i = 0; i < iterations; i++) {
      // Train the model using the data.
      const response = await this.model.fit(inputs, outputs, {
        epochs: 100,
        shuffle: true
      });

      this.setState({
        loop: i,
        loss: response.history.loss[0]
      });

      // If the loss is less than 0.001 lets just end
      if (response.history.loss[0] < 0.001) {
        break;
      }
    }
  }

  componentDidUpdate(prevProps, prevState) {
    if (
      prevState.has_red !== this.state.has_red ||
      prevState.has_white !== this.state.has_white ||
      prevState.has_yellow !== this.state.has_yellow
    ) {
      this.makePrediction();
    }
  }

  async makePrediction() {
    const input = tf.tensor2d([
      [this.state.has_red, this.state.has_white, this.state.has_yellow]
    ]);
    let prediction = await this.model.predict(input).dataSync();
    let colourPrediction = "White";

    if (prediction[0] > 0.8) {
      colourPrediction = "Pink";
    } else if (prediction[1] > 0.8) {
      colourPrediction = "Red";
    } else if (prediction[2] > 0.8) {
      colourPrediction = "White";
    } else if (prediction[3] > 0.8) {
      colourPrediction = "Green";
    } else if (prediction[4] > 0.8) {
      colourPrediction = "Yellow";
    }

    this.setState({
      prediction: colourPrediction
    });
  }

  handleChange(event) {
    this.setState({
      [event.target.name]: event.target.value === "true" ? 1 : 0
    });
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">Prediction</h1>
        </header>
        <p className="App-intro">Iteration: {this.state.loop}</p>
        <p className="App-intro">Loss: {this.state.loss}</p>
        <fieldset>
          <legend>Has Red?</legend>
          <label>
            <input
              type="radio"
              name="has_red"
              value="true"
              onChange={this.handleChange.bind(this)}
            />
            Yes
          </label>
          <label>
            <input
              type="radio"
              name="has_red"
              value="false"
              onChange={this.handleChange.bind(this)}
            />
            No
          </label>
        </fieldset>
        <fieldset>
          <legend>Has White?</legend>
          <label>
            <input
              type="radio"
              name="has_white"
              value="true"
              onChange={this.handleChange.bind(this)}
            />
            Yes
          </label>
          <label>
            <input
              type="radio"
              name="has_white"
              value="false"
              onChange={this.handleChange.bind(this)}
            />
            No
          </label>
        </fieldset>
        <fieldset>
          <legend>Has Yellow?</legend>
          <label>
            <input
              type="radio"
              name="has_yellow"
              value="true"
              onChange={this.handleChange.bind(this)}
            />
            Yes
          </label>
          <label>
            <input
              type="radio"
              name="has_yellow"
              value="false"
              onChange={this.handleChange.bind(this)}
            />
            No
          </label>
        </fieldset>
        <p className="App-intro">Predition: {this.state.prediction}</p>
      </div>
    );
  }
}

export default App;
