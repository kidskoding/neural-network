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
    Layer layer(3, 2); // 3 inputs, 2 neurons
    std::vector inputs = {1.0, 2.0, 3.0};

    layer.neurons[0].weights = {0.1, 0.2, 0.3};
    layer.neurons[0].bias = 0.1;
    layer.neurons[1].weights = {0.4, 0.5, 0.6};
    layer.neurons[1].bias = 0.2;

    for(auto& neuron : layer.neurons) {
        neuron.inputs = inputs;
    }

    layer.forward_propagation();

    double expected_output_0 = 0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0 + 0.1;
    double expected_output_1 = 0.4 * 1.0 + 0.5 * 2.0 + 0.6 * 3.0 + 0.2;

    ASSERT_EQ(layer.neurons.size(), 3);
    EXPECT_NEAR(layer.neurons[0].output, expected_output_0, 1e-4);
    EXPECT_NEAR(layer.neurons[1].output, expected_output_1, 1e-4);
}

TEST(LayerTest, Backpropagation) {
    Layer layer(3, 2);
    std::vector inputs = {1.0, 2.0, 3.0};
    std::vector actual = {0.5, 0.5};

    layer.neurons[0].weights = {0.1, 0.2, 0.3};
    layer.neurons[0].bias = 0.1;
    layer.neurons[1].weights = {0.4, 0.5, 0.6};
    layer.neurons[1].bias = 0.2;

    for(auto& neuron : layer.neurons) {
        neuron.inputs = inputs;
    }

    layer.forward_propagation();
    layer.backpropagation(actual);

    double error_0 = layer.neurons[0].output - actual[0];
    double gradient_0 = error_0 * (layer.neurons[0].output > 0 ? 1 : 0);
    double expected_weight_0_0 = 0.1 - 0.01 * gradient_0 * 1.0;
    double expected_weight_0_1 = 0.2 - 0.01 * gradient_0 * 2.0;
    double expected_weight_0_2 = 0.3 - 0.01 * gradient_0 * 3.0;
    double expected_bias_0 = 0.1 - 0.01 * gradient_0;

    double error_1 = layer.neurons[1].output - actual[1];
    double gradient_1 = error_1 * (layer.neurons[1].output > 0 ? 1 : 0);
    double expected_weight_1_0 = 0.4 - 0.01 * gradient_1 * 1.0;
    double expected_weight_1_1 = 0.5 - 0.01 * gradient_1 * 2.0;
    double expected_weight_1_2 = 0.6 - 0.01 * gradient_1 * 3.0;
    double expected_bias_1 = 0.2 - 0.01 * gradient_1;

    EXPECT_NEAR(layer.neurons[0].weights[0], expected_weight_0_0, 1e-4);
    EXPECT_NEAR(layer.neurons[0].weights[1], expected_weight_0_1, 1e-4);
    EXPECT_NEAR(layer.neurons[0].weights[2], expected_weight_0_2, 1e-4);
    EXPECT_NEAR(layer.neurons[0].bias, expected_bias_0, 1e-4);

    EXPECT_NEAR(layer.neurons[1].weights[0], expected_weight_1_0, 1e-4);
    EXPECT_NEAR(layer.neurons[1].weights[1], expected_weight_1_1, 1e-4);
    EXPECT_NEAR(layer.neurons[1].weights[2], expected_weight_1_2, 1e-4);
    EXPECT_NEAR(layer.neurons[1].bias, expected_bias_1, 1e-4);
}