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
};

#endif //LAYER_H