#include <gtest/gtest.h>
#include <cassert>
#include <small-nn.h>

using Math::LinAlg::Matrix;
const double EPS = 1e-5;
const double EPS_BIG = 1e-1;

TEST(LinAlg, Mult) {
    std::cout << "Testing linear algebra" << std::endl;

    const int64_t WIDTH = 3;
    const int64_t HEIGHT = 5;
    Matrix<HEIGHT, WIDTH> m;
    for(int i = 0; i < HEIGHT; i ++) {
        for(int j = 0; j < WIDTH; j ++) {
            m[i][j] = i * 10 + j;
        }
    }
    std::cout << m << std::endl;

    const auto mult = m * 5;
    std::cout << mult << std::endl;

    for(int i = 0; i < HEIGHT; i ++) {
        for(int j = 0; j < WIDTH; j ++) {
            assert(abs(m[i][j] * 5 - mult[i][j]) <= EPS);
        }
    }
}

TEST(ML, Backpropagation) {
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
    const auto diff = final - output;
    std::cout << "Optimized " << final << " " << output << " " << diff << std::endl;

    const auto total_diff = diff[0][0] + diff[0][1];
    std::cout << total_diff << " " << diff[0][0] << " " << diff[0][1] << std::endl;

    assert(total_diff <= EPS_BIG);
}

TEST(ML, Train) {
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
    const auto SAMPLES = 5;
    for(size_t i = 1; i <= SAMPLES; i ++) {
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

    assert(finalTotalLoss <= EPS_BIG);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
