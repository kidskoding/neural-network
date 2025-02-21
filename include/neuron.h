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
    std::function<double(double)> activationFunction;

    Neuron(int numInputs, std::function<double(double)> activation)
        : inputs(numInputs), weights(numInputs), activationFunction(activation) { }
};

#endif //NEURON_H