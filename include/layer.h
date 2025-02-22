#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <vector>

class Layer {
public:
    std::vector<Neuron> neurons;
    Layer* previousLayer;
    Layer* nextLayer;

    Layer(int numNeurons, int numInputsPerNeuron, Layer* previousLayer = nullptr) {
        
        this->previousLayer = previousLayer;
        for(int i = 0; i < numNeurons; i++) {
            neurons.push_back(Neuron(numInputsPerNeuron));
        }
    }

    static double loss(std::vector<double>& predicted, std::vector<double>& actual);
    void forward_propagation();
    void backpropagation(const std::vector<double>& actual);
};

#endif //LAYER_H