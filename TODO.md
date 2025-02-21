# TODO - Neural Network

Focuses on intuitively understanding 
how neural networks work, with a strong 
emphasis on backpropagation, gradient descent, 
and mathematical intuition rather than advanced 
architectures.

- [ ] Neurons and Layers
  - Define an **input layer**, **hidden layers**, and an **output layer**
  - Each neuron is connected to all neurons in the next layer (**fully connected**)
- [ ] Forward Propagation
  - Compute weighted sums of inputs plus biases
  - Apply activation functions (e.g. sigmoid, ReLU)
- [ ] Activation Functions
  - Implement common activation functions
    - **Sigmoid** for binary classification
    - **ReLU** for hidden layers
    - **Softmax** for multi-class classification
  - Apply activation functions after the weighted sum calculation in forward propogation
- [ ] Loss Function
  - Compute the difference between the predicted output and the actual output
  - Example: **Mean Squared Error (MSE)** for regression or **Cross-Entropy Loss** for classification
- [ ] Backpropagation
  - Calculate gradients using the chain rule
  - Adjust weights and biases using gradient descent
- [ ] Gradient Descent
  - **Optimize weights** using the gradients computed from backpropagation
  - Start with **Stochastic Gradient Descent (SGD)**
  - Later experiment with **Adam** or **Momentum-based methods**
- [ ] Optimizer
  - Implement and experiment with **Stochastic Gradient Descent (SGD)**
  - Add support for **advanced optimizers** like **Adam** or **Momentum** to improve weight updates
- [ ] Training Loop
  - Implement a loop that runs through multiple epochs
  - Update weights and biases after each epoch using forward propagation, loss calculation, backpropagation, and gradient descent
  - Track performance (loss) during training to observe progress and convergence