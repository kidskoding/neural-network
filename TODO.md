# TODO - Neural Network

Focuses on intuitively understanding 
how neural networks work, with a strong 
emphasis on backpropagation, gradient descent, 
and mathematical intuition rather than advanced 
architectures.

- [ ] Neurons and Layers
  - Define an **input layer**, **hidden layers**, and an **output layer**
  - Each neuron is connected to all neurons in the next layer (**fully connected**) 
- [ ] Forward Propogation
  - Compute weighted sums of inputs plus biases
  - Apply activation functions (e.g. signmoid, ReLU)
- [ ] Loss Function
  - Compute the difference between the predicted output and the actual output
  - Example: **Mean Squared Error (MSE)** for regression or **Cross-Entropy Loss** for classification
- [ ] Backpropagation
  - Calculate gradients using the chain rule
  - Adjust weights and biases using gradient descent
- [ ] Gradient Descent
  - **Optimize weights** using the gradients computed from backpropagation
  - Start with **Stochastic Gradient Descent (SGD)** and later experiment with **Adam** or **Momentum-based methods**