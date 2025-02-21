#include <gtest/gtest.h>
#include "neuron.h"

double identity(double x) {
    return x;
}

TEST(NeuronTest, WeightedSumTest) {
    Neuron neuron(2, identity);

    neuron.inputs = {1.0, 2.0};
    neuron.weights = {0.5, 0.5};
    neuron.bias = 1.0;

    double result = neuron.weighted_sum();
    EXPECT_DOUBLE_EQ(result, 2.5);
}
TEST(NeuronTest, WeightedSumWithNegativeBiasTest) {
    Neuron neuron(3, identity);

    neuron.inputs = {1.0, 1.0, 1.0};
    neuron.weights = {1.0, 1.0, 1.0};
    neuron.bias = -2.0;

    double result = neuron.weighted_sum();
    EXPECT_DOUBLE_EQ(result, 1.0);
}