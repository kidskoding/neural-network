#include <layer.h>
#include <gtest/gtest.h>

TEST(NeuronTest, LossFunction) {
    std::vector predicted = {1.0, 2.0, 3.0};
    std::vector actual = {1.0, 2.0, 3.0};
    double result = Layer::loss(predicted, actual);
    EXPECT_DOUBLE_EQ(result, 0.0);
}
TEST(NeuronTest, LossFunctionWithDifferentValues) {
    std::vector predicted = {1.0, 2.0, 3.0};
    std::vector actual = {2.0, 3.0, 4.0};
    double result = Layer::loss(predicted, actual);
    EXPECT_DOUBLE_EQ(result, 1.0);
}
TEST(NeuronTest, LossFunctionWithNegativeValues) {
    std::vector predicted = {-1.0, -2.0, -3.0};
    std::vector actual = {-1.0, -2.0, -3.0};
    double result = Layer::loss(predicted, actual);
    EXPECT_DOUBLE_EQ(result, 0.0);
}
TEST(NeuronTest, LossFunctionWithMixedValues) {
    std::vector predicted = {1.0, -2.0, 3.0};
    std::vector actual = {-1.0, 2.0, -3.0};
    double result = Layer::loss(predicted, actual);
    EXPECT_NEAR(result, 18.6667, 1e-4);
}

TEST(LayerTest, ForwardPropagation) {
    Layer layer(3, 2);
    std::vector inputs = {1.0, 2.0, 3.0};

    layer.neurons[0].weights = {0.1, 0.2, 0.3};
    layer.neurons[0].bias = 0.1;
    layer.neurons[1].weights = {0.4, 0.5, 0.6};
    layer.neurons[1].bias = 0.2;

    layer.forward_propagation(inputs);

    double expected_output_0 = std::max(0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0 + 0.1, 0.0);
    double expected_output_1 = std::max(0.4 * 1.0 + 0.5 * 2.0 + 0.6 * 3.0 + 0.2, 0.0);

    ASSERT_EQ(layer.neurons.size(), 3);
    EXPECT_NEAR(layer.neurons[0].output, expected_output_0, 1e-4);
    EXPECT_NEAR(layer.neurons[1].output, expected_output_1, 1e-4);
}

TEST(LayerTest, Backpropagation) {
    Layer layer(3, 2);
    std::vector<double> inputs = {1.0, 2.0, 3.0};
    std::vector<double> actual = {0.5, 0.5};

    layer.neurons[0].weights = {0.1, 0.2, 0.3};
    layer.neurons[0].bias = 0.1;
    layer.neurons[1].weights = {0.4, 0.5, 0.6};
    layer.neurons[1].bias = 0.2;

    layer.forward_propagation(inputs);
    layer.backpropagation(actual);

    for(size_t i = 0; i < layer.neurons.size(); i++) {
        double error = layer.neurons[i].output - actual[i];
        double gradient = error * (layer.neurons[i].output > 0 ? 1 : 0);

        for(size_t j = 0; j < layer.neurons[i].weights.size(); j++) {
            double expected_weight = layer.neurons[i].weights[j] - 0.01 * gradient * inputs[j];
            EXPECT_NEAR(layer.neurons[i].weights[j], expected_weight, 1e-4);
        }

        double expected_bias = layer.neurons[i].bias - 0.01 * gradient;
        EXPECT_NEAR(layer.neurons[i].bias, expected_bias, 1e-4);
    }
}