#ifndef NEURON_H
#define NEURON_H

#include <functional>
#include <vector>

class Neuron {
public:
    std::vector<double> inputs;
    std::vector<double> weights;
    double bias;
    double output;
    std::function<double(double)> activation_function;

    Neuron(int num_inputs, std::function<double(double)> activation)
        : inputs(num_inputs), weights(num_inputs), bias(0.0), activation_function(activation) { }

    double weighted_sum();
    static double relu(double x);
};

#endif //NEURON_H