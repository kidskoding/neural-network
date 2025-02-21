#include "../include/neuron.h"

double Neuron::weighted_sum() {
    double sum = 0;
    for(int i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }
    return sum + bias;    
}