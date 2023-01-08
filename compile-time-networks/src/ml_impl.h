#pragma once

#include "my_math.h"
#include "ml.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <ostream>

namespace MachineLearning {

template<size_t N, size_t M>
DenseLayer<N, M>::DenseLayer() : weights(LinAlg::Matrix<N, M>::random()), deltas(0) { }

template<size_t N, size_t M>
typename DenseLayer<N, M>::Output DenseLayer<N, M>::forwardPropagate(
    const DenseLayer<N, M>::Input &activation
) const {
    return activation * this->weights;
}

template<size_t N, size_t M>
template<typename Loss>
typename DenseLayer<N, M>::Input DenseLayer<N, M>::backPropagate(
    const DenseLayer<N, M>::Input &activation,
    const DenseLayer<N, M>::Output &outputDerivatives
) {
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            this->deltas[i][j] += activation[0][i] * outputDerivatives[0][j];
        }
    }
    return outputDerivatives * this->weights.transpose();
}

template<size_t N, size_t M>
void DenseLayer<N, M>::commitDeltas(const double deltaRatio) {
    this->weights = this->weights - this->deltas * deltaRatio;
    this->deltas = LinAlg::Matrix<N, M>(0);
}


template<size_t N>
BiasLayer<N>::BiasLayer() : bias(LinAlg::Matrix<1, N>::random()), deltas(0) { }

template<size_t N>
typename BiasLayer<N>::Output BiasLayer<N>::forwardPropagate(
    const typename BiasLayer<N>::Input &activation
) const {
    return activation + this->bias;
}

template<size_t N>
template<typename Loss>
typename BiasLayer<N>::Input BiasLayer<N>::backPropagate(
    const BiasLayer<N>::Input &activation,
    const BiasLayer<N>::Output &outputDerivatives
) {
    this->deltas = this->deltas + outputDerivatives;
    return outputDerivatives;
}

template<size_t N>
void BiasLayer<N>::commitDeltas(const double deltaRatio) {
    this->bias = this->bias - this->deltas * deltaRatio;
    this->deltas = LinAlg::Matrix<1, N>(0);
}


template<size_t N, typename F>
typename ApplicationLayer<N, F>::Output ApplicationLayer<N, F>::forwardPropagate(
    const typename ApplicationLayer<N, F>::Input &activation
) const {
    return activation.template apply<F>();
}

template<size_t N, typename F>
template<typename Loss>
typename ApplicationLayer<N, F>::Input ApplicationLayer<N, F>::backPropagate(
    const ApplicationLayer<N, F>::Input &activation,
    const ApplicationLayer<N, F>::Output &outputDerivatives
) {
    auto ret = outputDerivatives;
    for(size_t i = 0; i < N; i ++) {
        ret[0][i] *= F::derivative(activation[0][i]);
    }
    return ret;
}

template<size_t N, typename F>
void ApplicationLayer<N, F>::commitDeltas(const double deltaRatio) { }


template<typename FirstLayer, typename ...Rest>
NeuralNetwork<FirstLayer, Rest...>::NeuralNetwork() : firstLayer(), restNetwork() {
    static_assert(
        std::is_same<
            typename FirstLayer::Output,
            typename NeuralNetwork<Rest...>::Input
        >(),
        " Check if types in the neural network are concistent"
    );
}

template<typename FirstLayer, typename ...Rest>
typename NeuralNetwork<FirstLayer, Rest...>::Output NeuralNetwork<FirstLayer, Rest...>::forwardPropagate(
    const NeuralNetwork<FirstLayer, Rest...>::Input &input
) const {
    const auto ret = this->firstLayer.forwardPropagate(input);
    return this->restNetwork.forwardPropagate(ret);
}

template<typename FirstLayer, typename ...Rest>
template<typename Loss>
typename NeuralNetwork<FirstLayer, Rest...>::Input NeuralNetwork<FirstLayer, Rest...>::backPropagate(
    const NeuralNetwork<FirstLayer, Rest...>::Input &activation,
    const NeuralNetwork<FirstLayer, Rest...>::Output &target
) {
    const auto out = this->firstLayer.forwardPropagate(activation);
    const auto outputDerivatives = this->restNetwork.template backPropagate<Loss>(out, target);
    const auto ret = this->firstLayer.template backPropagate<Loss>(activation, outputDerivatives);
    return ret;
}

template<typename FirstLayer, typename ...Rest>
void NeuralNetwork<FirstLayer, Rest...>::commitDeltas(const double deltaRatio) {
    this->firstLayer.commitDeltas(deltaRatio);
    this->restNetwork.commitDeltas(deltaRatio);
}

template<typename FirstLayer>
typename NeuralNetwork<FirstLayer>::Output NeuralNetwork<FirstLayer>::forwardPropagate(
    const NeuralNetwork<FirstLayer>::Input &input
) const {
    const auto ret = this->firstLayer.forwardPropagate(input);
    return ret;
}

template<typename FirstLayer>
template<typename Loss>
typename NeuralNetwork<FirstLayer>::Input NeuralNetwork<FirstLayer>::backPropagate(
    const NeuralNetwork<FirstLayer>::Input &activation,
    const NeuralNetwork<FirstLayer>::Output &target
) {
    const auto out = this->firstLayer.forwardPropagate(activation);
    const auto outputDerivatives = LinAlg::zip<Loss::derivative>(out, target);
    const auto ret = this->firstLayer.template backPropagate<Loss>(activation, outputDerivatives);
    return ret;
}

template<typename FirstLayer>
void NeuralNetwork<FirstLayer>::commitDeltas(const double deltaRatio) {
    this->firstLayer.commitDeltas(deltaRatio);
}

template<typename Loss, typename Model>
Trainer<Loss, Model>::Trainer(Model *nn, const std::vector<typename Trainer<Loss, Model>::DataPoint> &data)
    : nn(nn), data(data) {}

template<typename Loss, typename Model>
void Trainer<Loss, Model>::runMinibatch(
    const std::pair<size_t, size_t> interval,
    const size_t iter,
    const double delta
) {
    for(size_t j = 0; j < iter; j ++) {
        for(size_t elem = interval.first; elem < interval.second; elem ++) {
            this->nn->template backPropagate<Loss>(this->data[elem].first, this->data[elem].second);
        }
        this->nn->commitDeltas(delta / (interval.second - interval.first));
    }
    const auto totalLoss = this->findTotalLoss();
    std::cout << "\r" << totalLoss << " for interval(" << interval.first << " " << interval.second << ")";
    std::flush(std::cout);
}

template<typename Loss, typename Model>
void Trainer<Loss, Model>::run(
    const size_t epochs,
    const size_t minibatchIter,
    const size_t batchSize,
    const double delta
) {
    for(size_t i = 0; i < epochs; i ++) {
        std::random_shuffle(this->data.begin(), this->data.end());
        for(size_t start = 0; start < this->data.size(); start += batchSize) {
            const auto end = std::min(start + batchSize, this->data.size());
            this->runMinibatch({start, end}, minibatchIter, delta);
        }
        std::cout << "\rIteration " << i << " Loss: " << this->findTotalLoss() << std::endl;
    }
}

template<typename Loss, typename Model>
double Trainer<Loss, Model>::findTotalLoss() const {
    double ret = 0;
    for(const auto &elem : this->data) {
        const auto out = this->nn->forwardPropagate(elem.first);
        const auto loss = LinAlg::zip<Loss::apply>(out, elem.second);
        for(size_t i = 0; i < Model::Output::SIZE_N; i ++) {
            for(size_t j = 0; j < Model::Output::SIZE_M; j ++) {
                ret += loss[i][j];
            }
        }
    }
    return ret;
}

};
