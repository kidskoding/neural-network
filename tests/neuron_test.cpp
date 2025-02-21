#include <gtest/gtest.h>
#include "neuron.h"

TEST(NeuronTest, WeightedSumTest) {
    Neuron neuron(2);

    neuron.inputs = {1.0, 2.0};
    neuron.weights = {0.5, 0.5};
    neuron.bias = 1.0;

    double result = neuron.weighted_sum();
    EXPECT_DOUBLE_EQ(result, 2.5);
}
TEST(NeuronTest, WeightedSumWithNegativeBiasTest) {
    Neuron neuron(3);

    neuron.inputs = {1.0, 1.0, 1.0};
    neuron.weights = {1.0, 1.0, 1.0};
    neuron.bias = -2.0;

    double result = neuron.weighted_sum();
    EXPECT_DOUBLE_EQ(result, 1.0);
}

TEST(NeuronTest, ReLUFunction) {
    Neuron neuron(3);
    EXPECT_EQ(neuron.relu(-1.0), 0.0);
    EXPECT_EQ(neuron.relu(0.0), 0.0);
    EXPECT_EQ(neuron.relu(1.0), 1.0);
}
TEST(NeuronTest, SigmoidFunction)
{
    Neuron neuron(3);
    EXPECT_DOUBLE_EQ(neuron.sigmoid(-1.0), 1.0 / (1.0 + std::exp(1.0)));
    EXPECT_DOUBLE_EQ(neuron.sigmoid(0.0), 1.0 / (1.0 + std::exp(0.0)));
    EXPECT_DOUBLE_EQ(neuron.sigmoid(1.0), 1.0 / (1.0 + std::exp(-1.0)));
}

TEST(NeuronTest, SoftmaxFunction) {
    Neuron neuron(3);

    std::vector<double> inputs = {1.0, 2.0, 3.0};
    std::vector<double> expected_output = {0.0900306, 0.244728, 0.665241};

    std::vector<double> result = Neuron::softmax(inputs);

    ASSERT_EQ(result.size(), expected_output.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected_output[i], 1e-6);
    }
}
TEST(NeuronTest, SoftmaxFunctionWithNegativeValues) {
    Neuron neuron(3);

    std::vector<double> inputs = {-1.0, -2.0, -3.0};
    std::vector<double> expected_output = {0.665241, 0.244728, 0.0900306};

    std::vector<double> result = Neuron::softmax(inputs);

    ASSERT_EQ(result.size(), expected_output.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected_output[i], 1e-6);
    }
}
TEST(NeuronTest, SoftmaxFunctionWithZeros) {
    Neuron neuron(3);

    std::vector<double> inputs = {0.0, 0.0, 0.0};
    std::vector<double> expected_output = {0.333333, 0.333333, 0.333333};

    std::vector<double> result = Neuron::softmax(inputs);

    ASSERT_EQ(result.size(), expected_output.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected_output[i], 1e-6);
    }
}

TEST(NeuronTest, LossFunction) {
    std::vector<double> predicted = {1.0, 2.0, 3.0};
    std::vector<double> actual = {1.0, 2.0, 3.0};
    double result = Neuron::loss(predicted, actual);
    EXPECT_DOUBLE_EQ(result, 0.0);
}
TEST(NeuronTest, LossFunctionWithDifferentValues) {
    std::vector<double> predicted = {1.0, 2.0, 3.0};
    std::vector<double> actual = {2.0, 3.0, 4.0};
    double result = Neuron::loss(predicted, actual);
    EXPECT_DOUBLE_EQ(result, 1.0);
}
TEST(NeuronTest, LossFunctionWithNegativeValues) {
    std::vector<double> predicted = {-1.0, -2.0, -3.0};
    std::vector<double> actual = {-1.0, -2.0, -3.0};
    double result = Neuron::loss(predicted, actual);
    EXPECT_DOUBLE_EQ(result, 0.0);
}
TEST(NeuronTest, LossFunctionWithMixedValues) {
    std::vector<double> predicted = {1.0, -2.0, 3.0};
    std::vector<double> actual = {-1.0, 2.0, -3.0};
    double result = Neuron::loss(predicted, actual);
    EXPECT_NEAR(result, 18.6667, 1e-4);
}