#include "../include/layer.h"

double Layer::loss(std::vector<double>& predicted, std::vector<double>& actual) {
    double sum = 0.0;
    for(size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - actual[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

void Layer::forward_propagation() {
    for(auto& n : neurons) {
        n.output = Neuron::relu(n.weighted_sum());
    }
}

void Layer::backpropagation(const std::vector<double>& actual) {
    for(size_t i = 0; i < neurons.size(); i++) {
        double error = neurons[i].output - actual[i];
        double gradient = error * (neurons[i].output > 0 ? 1 : 0);

        for(size_t j = 0; j < neurons[i].weights.size(); ++j) {
            neurons[i].weights[j] -= 0.01 * gradient * neurons[i].inputs[j];
        }
        neurons[i].bias -= 0.01 * gradient;
    }
}