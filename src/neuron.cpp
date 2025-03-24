#include "neuron.h"
#include <numeric>

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
std::vector<double> Neuron::softmax(std::vector<double>& x) {
    std::vector<double> exp_values(x.size());
    double max_value = *std::max_element(x.begin(), x.end());

    for(size_t i = 0; i < x.size(); i++) {
        exp_values[i] = std::exp(x[i] - max_value);
    }

    double sum_exp_values = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);
    for(size_t i = 0; i < exp_values.size(); i++) {
        exp_values[i] /= sum_exp_values;
    }

    return exp_values;
}
