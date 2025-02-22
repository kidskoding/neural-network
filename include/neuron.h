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

    Neuron(int num_inputs)
        : inputs(num_inputs), weights(num_inputs), bias(0.0) { }

    double weighted_sum();
    
    static double relu(double x);
    static double sigmoid(double x);
    static std::vector<double> softmax(std::vector<double>& x);
};

#endif //NEURON_H