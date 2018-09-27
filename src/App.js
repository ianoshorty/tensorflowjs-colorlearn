import React, { Component } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";

class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      prediction: null,
      loss: null,
      loop: 0
    };

    this.setupModel();
    this.trainModel(100);
  }

  setupModel() {
    // Define a model for linear regression.
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 4, inputShape: [2], activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1 }));
    
    // Prepare the model for training: Specify the loss and the optimizer.
    this.model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
  }

  async trainModel(iterations = 1) {
    
    // Generate some synthetic data for training.
    const inputs = tf.tensor2d([[1, 1], [1, 0], [0, 1], [0, 0]]);
    const outputs = tf.tensor2d([[1], [0], [0], [0]]);

    for (let i = 0; i < iterations; i++) {
      // Train the model using the data.
      const response = await this.model.fit(inputs, outputs, {
        epochs: 100,
        shuffle: true,
      });
      let prediction = await this.model.predict(tf.tensor2d([[0, 1]])).data()
      this.setState({
        loop: i,
        loss: response.history.loss[0],
        prediction
      });
    }    
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <h1 className="App-title">Prediction</h1>
        </header>
        <p className="App-intro">Iteration: {this.state.loop}</p>
        <p className="App-intro">Loss: {this.state.loss}</p>
        <p className="App-intro">Predition: {this.state.prediction}</p>
      </div>
    );
  }
}

export default App;
