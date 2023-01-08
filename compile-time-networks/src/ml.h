#pragma once

#include "my_math.h"
#include <iostream>

namespace MachineLearning {

template<size_t N, size_t M>
struct DenseLayer {
    typedef LinAlg::Matrix<1, N> Input;
    typedef LinAlg::Matrix<1, M> Output;

    LinAlg::Matrix<N, M> weights;
    LinAlg::Matrix<N, M> deltas;

    DenseLayer();
    Output forwardPropagate(const Input &activation) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &outputDerivative);
    void commitDeltas(const double deltaRatio);
};

template<size_t N>
struct BiasLayer {
    typedef LinAlg::Matrix<1, N> Input;
    typedef LinAlg::Matrix<1, N> Output;

    LinAlg::Matrix<1, N> bias;
    LinAlg::Matrix<1, N> deltas;

    BiasLayer();
    Output forwardPropagate(const Input &activation) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &outputDerivative);
    void commitDeltas(const double deltaRatio);
};

template<size_t N, typename F>
struct ApplicationLayer {
    typedef LinAlg::Matrix<1, N> Input;
    typedef LinAlg::Matrix<1, N> Output;

    ApplicationLayer() = default;
    Output forwardPropagate(const Input &activation) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &outputDerivative);
    void commitDeltas(const double deltaRatio);
};

template<typename FirstLayer, typename ...Rest>
struct NeuralNetwork {
    typedef typename FirstLayer::Input Input;
    typedef typename NeuralNetwork<Rest...>::Output Output;

    FirstLayer firstLayer;
    NeuralNetwork<Rest...> restNetwork;

    NeuralNetwork();
    Output forwardPropagate(const Input &input) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &target);
    void commitDeltas(const double deltaRatio);
};

template<typename FirstLayer> struct NeuralNetwork<FirstLayer> {
    typedef typename FirstLayer::Input Input;
    typedef typename FirstLayer::Output Output;

    FirstLayer firstLayer;

    NeuralNetwork() = default;
    Output forwardPropagate(const Input &input) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &target);
    void commitDeltas(const double deltaRatio);
};

template<typename Loss, typename Model>
struct Trainer {
    typedef std::pair<typename Model::Input, typename Model::Output> DataPoint;

    std::vector<DataPoint> data;
    Model *nn;

    Trainer(Model *nn, const std::vector<DataPoint> &data);
    void run(
        const size_t epochs,
        const size_t minibatchIter,
        const size_t batchSize,
        const double delta = 0.01
    );
    void runMinibatch(
        const std::pair<size_t, size_t> interval,
        const size_t iter,
        const double delta = 0.01
    );
    double findTotalLoss() const;
};

};

