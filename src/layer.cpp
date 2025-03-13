#include "../include/layer.h"

//test comment for profile README
double Layer::loss(std::vector<double>& predicted, std::vector<double>& actual) {
    double sum = 0.0;
    for(size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - actual[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

void Layer::forward_propagation(const std::vector<double>& inputs) {
    for(auto& neuron : neurons) {
        neuron.inputs = inputs;
        double sum = neuron.bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * neuron.weights[i];
        }
        neuron.output = Neuron::relu(sum);
    }
}

void Layer::backpropagation(const std::vector<double>& actual) {
    double learning_rate = 0.01;
    for(size_t i = 0; i < neurons.size(); ++i) {
        double error = neurons[i].output - actual[i];
        double gradient = error * (neurons[i].output > 0 ? 1 : 0);

        for(size_t j = 0; j < neurons[i].weights.size(); ++j) {
            neurons[i].weights[j] -= learning_rate * gradient * neurons[i].inputs[j];
        }
        neurons[i].bias -= learning_rate * gradient;
    }
}
