#pragma once

#include <iostream>
#include <ostream>
#include <ranges>
#include <math.h>
#include <cstring>
#include <vector>
#include <stdio.h>

#include "my_math.h"

namespace Continuous {

inline double Sigmoid::apply(const double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double Sigmoid::derivative(const double x) {
    return Sigmoid::apply(x) * (1.0 - Sigmoid::apply(x));
}

inline double Relu::apply(const double x) {
    return (x >= 0) ? x : 0;
}

inline double Relu::derivative(const double x) {
    return (x >= 0) ? 1 : 0;
}

inline double LeakyRelu::apply(const double x) {
    return (x >= 0) ? x : x * 0.01;
}

inline double LeakyRelu::derivative(const double x) {
    return (x >= 0) ? 1 : 0.01;
}

template<int C>
double Adder<C>::apply(const double x) {
    return x + C;
}

template<int32_t C>
double Mult<C>::apply(const double x) {
    return x * C;
}

inline double QuadraticLoss::apply(const double x, const double y) {
    return (x - y) * (x - y);
}
inline double QuadraticLoss::derivative(const double x, const double y) {
    return 2 * x - 2 * y;
}

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
            this->data[i][j] = value;
        }
    }
}

template<size_t N, size_t M>
Matrix<N, M>::Matrix(const Matrix<N, M> &oth) {
    memcpy(this->data, oth.data, sizeof(this->data));
}

template<size_t N, size_t M>
double* Matrix<N, M>::operator[](const size_t &i) {
    return this->data[i];
}

template<size_t N, size_t M>
const double* Matrix<N, M>::operator[](const size_t &i) const {
    return this->data[i];
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
    memcpy(this->data, oth.data, sizeof(this->data));
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
