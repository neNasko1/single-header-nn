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

void testML() {
    std::cout << "Testing machine learning" << std::endl;

    LinAlg::Matrix<1, 3> m;
    MachineLearning::ApplicationLayer<3, Continuous::Adder<10> > f;
    std::cout << f.forwardPropagate(m) << std::endl;

    #define FullyConnectedApplicationLayer(N, M, F) MachineLearning::DenseLayer<N, M>, MachineLearning::BiasLayer<M>, MachineLearning::ApplicationLayer<M, F>
    MachineLearning::NeuralNetwork<
        FullyConnectedApplicationLayer(3, 10, Continuous::Sigmoid),
        FullyConnectedApplicationLayer(10, 1, Continuous::Relu)
    > nn;
    std::cout << nn.forwardPropagate(m) << std::endl;

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
    trainer.run(10000, 10, 10, 0.0005);
    assert(trainer.findTotalLoss() <= 0.01);
}

int main() {
    testLinAlg();
    testML();
}
