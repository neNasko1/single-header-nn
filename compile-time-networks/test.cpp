#include <iostream>
#include "src/small_nn.h"
#include <cassert>

void testLinAlg() {
    std::cout << "Testing linear algebra" << std::endl;

    LinAlg::Matrix<3, 3> m;
    for(int i = 0; i < 3; i ++) {
        for(int j = 0; j < 3; j ++) {
            m[i][j] = i * 10 + j;
        }
    }
    std::cout << m << std::endl;

    const auto ad = m.apply<Continuous::Adder<5> >();
    std::cout << ad << std::endl;

    const auto mult = m * ad;
    std::cout << mult << std::endl;

    const auto mult_scalar = mult.apply<Continuous::Mult<2>>();
    std::cout << mult_scalar << std::endl;
}

void testMLTraining() {
    std::cout << "Started testing full training" << std::endl;

    #define FullyConnectedApplicationLayer(N, M, F) MachineLearning::DenseLayer<N, M>, MachineLearning::BiasLayer<M>, MachineLearning::ApplicationLayer<M, F>
    MachineLearning::NeuralNetwork<
        FullyConnectedApplicationLayer(3, 10, Continuous::Sigmoid),
        FullyConnectedApplicationLayer(10, 1, Continuous::Sigmoid)
    > nn;

    std::vector<std::pair<LinAlg::Matrix<1, 3>, LinAlg::Matrix<1, 1> > > dataPoints;
    const auto SAMPLES = 10;
    for(size_t i = 1; i < SAMPLES; i ++) {
        LinAlg::Matrix<1, 3> input;
        LinAlg::Matrix<1, 1> output;

        for(size_t j = 0; j < 3; j ++) { input[0][j] = (j + 1) * i; }
        output[0][0] = (double)i / SAMPLES;

        dataPoints.push_back({input, output});
    }

    MachineLearning::Trainer<Continuous::QuadraticLoss, decltype(nn)> trainer(&nn, dataPoints);
    trainer.run(1000, 10, dataPoints.size(), 0.05);
    const auto finalTotalLoss = trainer.findTotalLoss();
    std::cout << finalTotalLoss << std::endl;
    assert(finalTotalLoss <= 0.01);
}

void testMLBackpropagation() {
    std::cout << "Started testing backpropagation" << std::endl;

    #define FullyConnectedApplicationLayer(N, M, F) MachineLearning::DenseLayer<N, M>, MachineLearning::BiasLayer<M>, MachineLearning::ApplicationLayer<M, F>
    MachineLearning::NeuralNetwork<
        FullyConnectedApplicationLayer(3, 2, Continuous::Sigmoid)
    > nn;

    LinAlg::Matrix<1, 3> input; input[0][0] = 0.1; input[0][1] = -0.5; input[0][2] = 0.6;
    LinAlg::Matrix<1, 2> output; output[0][0] = 0.5; output[0][1] = 0.2;

    const auto initial = nn.forwardPropagate(input);

    for(size_t i = 0; i < 1000; i ++) {
        nn.backPropagate<Continuous::QuadraticLoss>(input, output);
        const auto whox = rand() % 3, whoy = rand() % 2;
        nn.firstLayer.weights[whox][whoy] -= nn.firstLayer.deltas[whox][whoy];
        nn.firstLayer.deltas = LinAlg::Matrix<3, 2>(0);
    }

    const auto final = nn.forwardPropagate(input);
    std::cout << "Optimized " << final << " " << output << std::endl;
}

void testML() {
    std::cout << "Testing machine learning" << std::endl;

    LinAlg::Matrix<1, 3> m;
    MachineLearning::ApplicationLayer<3, Continuous::Adder<10> > f;
    std::cout << f.forwardPropagate(m) << std::endl;

    testMLTraining();
    testMLBackpropagation();
}

int main() {
    testLinAlg();
    testML();
}
