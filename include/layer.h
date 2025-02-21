#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include <vector>

class Layer {
public:
    std::vector<Neuron> neurons;
    Layer* previousLayer;
    Layer* nextLayer;

    Layer(int numNeurons, int numInputsPerNeuron,
        std::function<double(double)> activationFn, Layer* previousLayer = nullptr) {
        
        this->previousLayer = previousLayer;
        for(int i = 0; i < numNeurons; i++) {
            neurons.push_back(Neuron(numInputsPerNeuron, activationFn));
        }
    }
};

#endif //LAYER_H