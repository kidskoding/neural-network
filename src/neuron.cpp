#include "../include/neuron.h"

double Neuron::weighted_sum() {
    double sum = 0;
    for(int i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    return sum + bias;    
}

double Neuron::relu(double x) {
    return std::max(0.0, x);
}
double Neuron::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
