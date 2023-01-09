#include "small-nn.h"
#include <iostream>
#include <cassert>

using Math::LinAlg::Matrix;

void testLinAlg() {
    std::cout << "Testing linear algebra" << std::endl;

    Matrix<3, 3> m;
    for(int i = 0; i < 3; i ++) {
        for(int j = 0; j < 3; j ++) {
            m[i][j] = i * 10 + j;
        }
    }
    std::cout << m << std::endl;

    const auto mult = m * 5;
    std::cout << mult << std::endl;
}

void testMLTraining() {
    std::cout << "Started testing full training" << std::endl;

    #define FullyConnectedApplicationLayer(N, M, F) \
        MachineLearning::Layer::DenseLayer<N, M>, \
        MachineLearning::Layer::BiasLayer<M>, \
        MachineLearning::Layer::ApplicationLayer<M, F>
    MachineLearning::Model::NeuralNetwork<
        FullyConnectedApplicationLayer(3, 10, Math::Continuous::Sigmoid),
        FullyConnectedApplicationLayer(10, 1, Math::Continuous::Sigmoid)
    > nn;

    std::vector<std::pair<Matrix<1, 3>, Matrix<1, 1> > > dataPoints;
    const auto SAMPLES = 10;
    for(size_t i = 1; i < SAMPLES; i ++) {
        Matrix<1, 3> input;
        Matrix<1, 1> output;

        for(size_t j = 0; j < 3; j ++) { input[0][j] = (j + 1) * i; }
        output[0][0] = (double)i / SAMPLES;

        dataPoints.push_back({input, output});
    }

    MachineLearning::Training::Trainer<Math::Continuous::QuadraticLoss, decltype(nn)> trainer(&nn, dataPoints);
    trainer.run(1000, 10, dataPoints.size(), 0.05);
    const auto finalTotalLoss = trainer.findTotalLoss();
    std::cout << finalTotalLoss << std::endl;
    assert(finalTotalLoss <= 0.01);
}

void testMLBackpropagation() {
    std::cout << "Started testing backpropagation" << std::endl;

    #define FullyConnectedApplicationLayer(N, M, F) \
        MachineLearning::Layer::DenseLayer<N, M>, \
        MachineLearning::Layer::BiasLayer<M>, \
        MachineLearning::Layer::ApplicationLayer<M, F>
    MachineLearning::Model::NeuralNetwork<
        FullyConnectedApplicationLayer(3, 2, Math::Continuous::Sigmoid)
    > nn;

    Matrix<1, 3> input; input[0][0] = 0.1; input[0][1] = -0.5; input[0][2] = 0.6;
    Matrix<1, 2> output; output[0][0] = 0.5; output[0][1] = 0.2;

    const auto initial = nn.forwardPropagate(input);

    for(size_t i = 0; i < 1000; i ++) {
        nn.backPropagate<Math::Continuous::QuadraticLoss>(input, output);
        const auto whox = rand() % 3, whoy = rand() % 2;
        nn.firstLayer.weights[whox][whoy] -= nn.firstLayer.deltas[whox][whoy];
        nn.firstLayer.deltas = Matrix<3, 2>(0);
    }

    const auto final = nn.forwardPropagate(input);
    std::cout << "Optimized " << final << " " << output << std::endl;
}

void testML() {
    std::cout << "Testing machine learning" << std::endl;

    testMLTraining();
    testMLBackpropagation();
}

int main() {
    testLinAlg();
    testML();
}
