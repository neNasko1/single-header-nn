#pragma once

#include <iostream>
#include <ostream>
#include <random>
#include <ranges>
#include <math.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdio.h>
#include <algorithm>

namespace Math {

namespace Continuous {

template<typename T>
struct Function {
    static T apply(const T x);
    static T derivative(const T x);
};

template<typename T>
struct DyadicFunction {
    static T apply(const T x, const T y);
    static T derivative(const T x, const T y);
};

struct Sigmoid : public Function<double> {
    static double apply(const double x);
    static double derivative(const double x);
};

struct Relu : public Function<double> {
    static double apply(const double x);
    static double derivative(const double x);
};

struct LeakyRelu : public Function<double> {
    static double apply(const double x);
    static double derivative(const double x);
};

struct QuadraticLoss : public DyadicFunction<double> {
    static double apply(const double x, const double y);
    static double derivative(const double x, const double y);
};

template<int32_t MIN, int32_t MAX>
double randomValue();

};

namespace LinAlg {

template<size_t N, size_t M>
struct Matrix {
    static constexpr auto SIZE_N = N;
    static constexpr auto SIZE_M = M;

    static Matrix<N, M> random();

    double data[N][M];

    Matrix(const double value = 0);

    double* operator[](const size_t &i);
    const double* operator[](const size_t &i) const;

    template<typename F>
    Matrix<N, M> apply() const;

    Matrix<N, M> operator+(const Matrix<N, M> &oth) const;
    Matrix<N, M> operator-(const Matrix<N, M> &oth) const;
    template<size_t K>
    Matrix<N, K> operator*(const Matrix<M, K> &oth) const;
    Matrix<N, M> operator*(const double &oth) const;
    Matrix<N, M> operator=(const Matrix<N, M> &oth);
    Matrix<M, N> transpose() const;
};

template<double F(const double, const double), size_t N, size_t M>
Matrix<N, M> zip(const Matrix<N, M> &a, const Matrix <N, M> &b);

template<size_t N, size_t M>
std::ostream& operator<<(std::ostream &out, const Matrix<N, M> &mat);

};

};


namespace Math { // implementation

namespace Continuous {

inline double Sigmoid::apply(const double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double Sigmoid::derivative(const double x) { return Sigmoid::apply(x) * (1.0 - Sigmoid::apply(x)); }

inline double Relu::apply(const double x) { return (x >= 0) ? x : 0; }
inline double Relu::derivative(const double x) { return (x >= 0) ? 1 : 0; }

inline double LeakyRelu::apply(const double x) { return (x >= 0) ? x : x * 0.01; }

inline double LeakyRelu::derivative(const double x) { return (x >= 0) ? 1 : 0.01; }

inline double QuadraticLoss::apply(const double x, const double y) { return (x - y) * (x - y); }
inline double QuadraticLoss::derivative(const double x, const double y) { return 2 * x - 2 * y; }

template<int32_t MIN, int32_t MAX>
double randomValue() {
    static_assert(MIN <= MAX, "Random value expects MIN <= MAX");
    const auto ret = MIN + ((double)rand()) / (double)RAND_MAX * (MAX - MIN);
    return ret;
}

};

namespace LinAlg {

template<size_t N, size_t M>
Matrix<N, M> Matrix<N, M>::random()  {
    Matrix<N, M> ret;
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            ret[i][j] = Continuous::randomValue<-1, 1>();
        }
    }
    return ret;
}

template<size_t N, size_t M>
Matrix<N, M>::Matrix(const double value) {
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            data[i][j] = value;
        }
    }
}

template<size_t N, size_t M>
double* Matrix<N, M>::operator[](const size_t &i) {
    return data[i];
}

template<size_t N, size_t M>
const double* Matrix<N, M>::operator[](const size_t &i) const {
    return data[i];
}

template<size_t N, size_t M>
template<typename F>
Matrix<N, M> Matrix<N, M>::apply() const {
    static_assert(
        std::is_base_of<Continuous::Function<double>, F>::value,
        "Matrix::apply was not provided with a Continuous::Function<double>"
    );

    Matrix<N, M> ret;
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            ret[i][j] = F::apply((*this)[i][j]);
        }
    }
    return ret;
}

template<size_t N, size_t M>
Matrix<N, M> Matrix<N, M>::operator+(const Matrix<N, M> &oth) const {
    Matrix<N, M> ret;
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            ret[i][j] = (*this)[i][j] + oth[i][j];
        }
    }
    return ret;
}

template<size_t N, size_t M>
Matrix<N, M> Matrix<N, M>::operator-(const Matrix<N, M> &oth) const {
    Matrix<N, M> ret;
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            ret[i][j] = (*this)[i][j] - oth[i][j];
        }
    }
    return ret;
}

template<size_t N, size_t M>
Matrix<N, M> Matrix<N, M>::operator=(const Matrix<N, M> &oth) {
    memcpy(data, oth.data, sizeof(data));
    return *this;
}

template<size_t N, size_t M>
template<size_t K>
Matrix<N, K> Matrix<N, M>::operator*(const Matrix<M, K> &oth) const {
    Matrix<N, K> ret(0);
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            for(size_t k = 0; k < K; k ++) {
                ret[i][k] += (*this)[i][j] * oth[j][k];
            }
        }
    }
    return ret;
}

template<size_t N, size_t M>
Matrix<N, M> Matrix<N, M>::operator*(const double &oth) const {
    Matrix<N, M> ret;
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            ret[i][j] = (*this)[i][j] * oth;
        }
    }
    return ret;
}

template<size_t N, size_t M>
Matrix<M, N> Matrix<N, M>::transpose() const {
    Matrix<M, N> ret;
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            ret[j][i] = (*this)[i][j];
        }
    }
    return ret;
}


template<double F(const double, const double), size_t N, size_t M>
Matrix<N, M> zip(const Matrix<N, M> &a, const Matrix <N, M> &b) {
    Matrix<N, M> ret;
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            ret[i][j] = F(a[i][j], b[i][j]);
        }
    }
    return ret;
}

template<size_t N, size_t M>
std::ostream& operator<<(std::ostream &out, const Matrix<N, M> &mat) {
    if constexpr (N == 1) {
        out << "Matrix<" << N << ", " << M << "> { ";
        for(size_t j = 0; j < M; j ++) {
            out << mat[0][j] << " ";
        }
        out << "}";
        return out;
    }

    out << "Matrix<" << N << ", " << M << "> {" << std::endl;
    for(size_t i = 0; i < N; i ++) {
        out << "\t";
        for(size_t j = 0; j < M; j ++) {
            out << mat[i][j] << " ";
        }
        out << std::endl;
    }
    out << "} ";
    return out;
}

};

};


namespace MachineLearning {

namespace Layer {

template<size_t N, size_t M>
struct DenseLayer {
    typedef Math::LinAlg::Matrix<1, N> Input;
    typedef Math::LinAlg::Matrix<1, M> Output;

    Math::LinAlg::Matrix<N, M> weights;
    Math::LinAlg::Matrix<N, M> deltas;

    DenseLayer();
    Output forwardPropagate(const Input &activation) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &outputDerivative);
    void commitDeltas(const double deltaRatio);
};

template<size_t N>
struct BiasLayer {
    typedef Math::LinAlg::Matrix<1, N> Input;
    typedef Math::LinAlg::Matrix<1, N> Output;

    Math::LinAlg::Matrix<1, N> bias;
    Math::LinAlg::Matrix<1, N> deltas;

    BiasLayer();
    Output forwardPropagate(const Input &activation) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &outputDerivative);
    void commitDeltas(const double deltaRatio);
};

template<size_t N, typename F>
struct ApplicationLayer {
    typedef Math::LinAlg::Matrix<1, N> Input;
    typedef Math::LinAlg::Matrix<1, N> Output;

    ApplicationLayer() = default;
    Output forwardPropagate(const Input &activation) const;
    template<typename Loss>
    Input backPropagate(const Input &activation, const Output &outputDerivative);
    void commitDeltas(const double deltaRatio);
};

};

namespace Model {

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

};

namespace Training {

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

};


namespace MachineLearning { // implementation

namespace Layer {

template<size_t N, size_t M>
DenseLayer<N, M>::DenseLayer() : weights(Math::LinAlg::Matrix<N, M>::random()), deltas(0) { }

template<size_t N, size_t M>
typename DenseLayer<N, M>::Output DenseLayer<N, M>::forwardPropagate(
    const DenseLayer<N, M>::Input &activation
) const {
    return activation * weights;
}

template<size_t N, size_t M>
template<typename Loss>
typename DenseLayer<N, M>::Input DenseLayer<N, M>::backPropagate(
    const DenseLayer<N, M>::Input &activation,
    const DenseLayer<N, M>::Output &outputDerivatives
) {
    for(size_t i = 0; i < N; i ++) {
        for(size_t j = 0; j < M; j ++) {
            deltas[i][j] += activation[0][i] * outputDerivatives[0][j];
        }
    }
    return outputDerivatives * weights.transpose();
}

template<size_t N, size_t M>
void DenseLayer<N, M>::commitDeltas(const double deltaRatio) {
    weights = weights - deltas * deltaRatio;
    deltas = Math::LinAlg::Matrix<N, M>(0);
}

template<size_t N>
BiasLayer<N>::BiasLayer() : bias(Math::LinAlg::Matrix<1, N>::random()), deltas(0) { }

template<size_t N>
typename BiasLayer<N>::Output BiasLayer<N>::forwardPropagate(
    const typename BiasLayer<N>::Input &activation
) const {
    return activation + bias;
}

template<size_t N>
template<typename Loss>
typename BiasLayer<N>::Input BiasLayer<N>::backPropagate(
    const BiasLayer<N>::Input &activation,
    const BiasLayer<N>::Output &outputDerivatives
) {
    deltas = deltas + outputDerivatives;
    return outputDerivatives;
}

template<size_t N>
void BiasLayer<N>::commitDeltas(const double deltaRatio) {
    bias = bias - deltas * deltaRatio;
    deltas = Math::LinAlg::Matrix<1, N>(0);
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

};

namespace Model {

template<typename FirstLayer, typename ...Rest>
NeuralNetwork<FirstLayer, Rest...>::NeuralNetwork() : firstLayer(), restNetwork() {
    static_assert(
        std::is_same<
            typename FirstLayer::Output,
            typename NeuralNetwork<Rest...>::Input
        >(),
        " Check if types in the neural network are consistent"
    );
}

template<typename FirstLayer, typename ...Rest>
typename NeuralNetwork<FirstLayer, Rest...>::Output NeuralNetwork<FirstLayer, Rest...>::forwardPropagate(
    const NeuralNetwork<FirstLayer, Rest...>::Input &input
) const {
    const auto ret = firstLayer.forwardPropagate(input);
    return restNetwork.forwardPropagate(ret);
}

template<typename FirstLayer, typename ...Rest>
template<typename Loss>
typename NeuralNetwork<FirstLayer, Rest...>::Input NeuralNetwork<FirstLayer, Rest...>::backPropagate(
    const NeuralNetwork<FirstLayer, Rest...>::Input &activation,
    const NeuralNetwork<FirstLayer, Rest...>::Output &target
) {
    const auto out = firstLayer.forwardPropagate(activation);
    const auto outputDerivatives = restNetwork.template backPropagate<Loss>(out, target);
    const auto ret = firstLayer.template backPropagate<Loss>(activation, outputDerivatives);
    return ret;
}

template<typename FirstLayer, typename ...Rest>
void NeuralNetwork<FirstLayer, Rest...>::commitDeltas(const double deltaRatio) {
    firstLayer.commitDeltas(deltaRatio);
    restNetwork.commitDeltas(deltaRatio);
}

template<typename FirstLayer>
typename NeuralNetwork<FirstLayer>::Output NeuralNetwork<FirstLayer>::forwardPropagate(
    const NeuralNetwork<FirstLayer>::Input &input
) const {
    const auto ret = firstLayer.forwardPropagate(input);
    return ret;
}

template<typename FirstLayer>
template<typename Loss>
typename NeuralNetwork<FirstLayer>::Input NeuralNetwork<FirstLayer>::backPropagate(
    const NeuralNetwork<FirstLayer>::Input &activation,
    const NeuralNetwork<FirstLayer>::Output &target
) {
    const auto out = firstLayer.forwardPropagate(activation);
    const auto outputDerivatives = Math::LinAlg::zip<Loss::derivative>(out, target);
    const auto ret = firstLayer.template backPropagate<Loss>(activation, outputDerivatives);
    return ret;
}

template<typename FirstLayer>
void NeuralNetwork<FirstLayer>::commitDeltas(const double deltaRatio) {
    firstLayer.commitDeltas(deltaRatio);
}

};

namespace Training {

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
            nn->template backPropagate<Loss>(data[elem].first, data[elem].second);
        }
        nn->commitDeltas(delta / (interval.second - interval.first));
    }
    const auto totalLoss = findTotalLoss();
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
        std::shuffle(data.begin(), data.end(), std::mt19937{std::random_device{}()});
        for(size_t start = 0; start < data.size(); start += batchSize) {
            const auto end = std::min(start + batchSize, data.size());
            runMinibatch({start, end}, minibatchIter, delta);
        }
        std::cout << "\rIteration " << i << " Loss: " << findTotalLoss() << std::endl;
    }
}

template<typename Loss, typename Model>
double Trainer<Loss, Model>::findTotalLoss() const {
    double ret = 0;
    for(const auto &elem : data) {
        const auto out = nn->forwardPropagate(elem.first);
        const auto loss = Math::LinAlg::zip<Loss::apply>(out, elem.second);
        for(size_t i = 0; i < Model::Output::SIZE_N; i ++) {
            for(size_t j = 0; j < Model::Output::SIZE_M; j ++) {
                ret += loss[i][j];
            }
        }
    }
    return ret;
}

};

};

