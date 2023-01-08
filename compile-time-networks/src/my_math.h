#pragma once

#include <random>
#include <ranges>
#include <math.h>
#include <cstdint>
#include <vector>

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

template<int C>
struct Adder : public Function<double> {
    static double apply(const double x);
};

template<int32_t C>
struct Mult : public Function<double> {
    static double apply(const double x);
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

    double data[N][M];

    Matrix();
    Matrix(const double value);
    Matrix(const Matrix<N, M> &oth) = default;
    Matrix(Matrix<N, M> &&oth) = default;

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
