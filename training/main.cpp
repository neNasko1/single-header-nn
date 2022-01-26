#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <span>
#include <time.h>

namespace util {

// This random is actually really bad so it should be replaced with a better alternative
double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivative(double x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

};

struct Matrix {
    size_t N, M;
    std::vector<double> tab;

    Matrix(const size_t _N, const size_t _M) : N(_N), M(_M), tab(_N * _M, 0) { }

    Matrix(const Matrix &oth) : N(oth.N), M(oth.M), tab(oth.tab) { }

    double &operator ()(const size_t i, const size_t j) {
        return tab[i * M + j];
    }

    double get(const size_t i, const size_t j) const {
        return tab[i * M + j];
    }

    static Matrix randomMatrix(const size_t N, const size_t M) {
        Matrix ret(N, M);
        for(size_t i = 0; i < N; i ++) {
            for(size_t j = 0; j < M; j ++) {
                ret(i, j) = util::fRand(-1, 1);
            }
        }
        return ret;
    }
};

Matrix operator *(const Matrix &a, const Matrix &b) {
    Matrix ret(a.N, b.M);

#ifdef DEBUG
    assert(a.M == b.N);
#endif

    for(size_t i = 0; i < a.N; i ++) {
        for(size_t j = 0; j < b.M; j ++) {
            for(size_t k = 0; k < a.M; k ++) {
                ret(i, j) += a.get(i, k) * b.get(k, j);
            }
        }
    }

    return ret;
}

Matrix operator +(const Matrix &a, const Matrix &b) {
    Matrix ret(a.N, a.M);

#ifdef DEBUG
    assert(a.N == b.N && a.M == b.M);
#endif

    for(size_t i = 0; i < a.N; i ++) {
        for(size_t j = 0; j < b.M; j ++) {
            ret(i, j) = a.get(i, j) + b.get(i, j);
        }
    }

    return ret;
}

Matrix operator -(const Matrix &a, const Matrix &b) {
    Matrix ret(a.N, a.M);

#ifdef DEBUG
    assert(a.N == b.N && a.M == b.M);
#endif

    for(size_t i = 0; i < a.N; i ++) {
        for(size_t j = 0; j < b.M; j ++) {
            ret(i, j) = a.get(i, j) - b.get(i, j);
        }
    }

    return ret;
}

template<typename F>
Matrix apply(const Matrix &a, F f) {
    Matrix ret(a.N, a.M);

    for(size_t i = 0; i < a.N; i ++) {
        for(size_t j = 0; j < a.M; j ++) {
            ret(i, j) = f(a.get(i, j));
        }
    }

    return ret;
}

struct NeuralNetwork {
    std::vector<size_t> layerLength;
    std::vector<Matrix> weight, bias;

    NeuralNetwork(const std::vector<size_t> &_layerLength) : layerLength(_layerLength), weight(), bias() {
        for(size_t i = 0; i < layerLength.size() - 1; i ++) {
            weight.push_back(Matrix::randomMatrix(layerLength[i], layerLength[i + 1]));
        }
        for(size_t i = 1; i < layerLength.size(); i ++) {
            bias.push_back(Matrix::randomMatrix(1, layerLength[i]));
        }
    }

    Matrix forwardPropagate(const Matrix &input) const {
        Matrix ret = input;

        for(size_t i = 0; i < weight.size(); i ++) {
            ret = apply(ret * weight[i] + bias[i], util::sigmoid);
        }

        return ret;
    }

    double cost(const Matrix &input, const Matrix &output) const {
        Matrix dist = forwardPropagate(input) - output;

        double ret = 0;
        for(size_t i = 0; i < output.M; i ++) {
            ret += dist(0, i) * dist(0, i);
        }
        return ret;
    }

    void backPropagate(const Matrix &input, const Matrix &output, std::pair<std::vector<Matrix>, std::vector<Matrix> > &delta) {
        Matrix ret = input;
        std::vector<Matrix> activation = {ret};

        for(size_t i = 0; i < weight.size(); i ++) {
            ret * weight[i];
            ret = apply(ret * weight[i] + bias[i], util::sigmoid);
            activation.push_back(ret);
        }


        Matrix activationDelta = apply(ret - output, [](const double x) {return x * 2.;});

        for(int ind = weight.size() - 1; ind >= 0; ind --) {
            for(size_t i = 0; i < weight[ind].N; i ++) {
                for(size_t j = 0; j < weight[ind].M; j ++) {
                    delta.first[ind](i, j) += activation[ind](0, i) * util::sigmoidDerivative(activation[ind + 1](0, j)) * activationDelta(0, j);
                }
            }
            for(size_t i = 0; i < weight[ind].M; i ++) {
                delta.second[ind](0, i) += util::sigmoidDerivative(activation[ind + 1](0, i)) * activationDelta(0, i);
            }

            Matrix newActivationDelta = Matrix(1, weight[ind].N);
            for(size_t i = 0; i < weight[ind].N; i ++) {
                for(size_t j = 0; j < weight[ind].M; j ++) {
                    newActivationDelta(0, i) += weight[ind](i, j) * util::sigmoidDerivative(activation[ind + 1](0, j)) * activationDelta(0, j);
                }
            }
            activationDelta = newActivationDelta;
        }
    }
};

const int PIC_N = 28, PIC_M = 28;
const int LABEL_CNT = 10;

double getAcceptanceRate(NeuralNetwork &nn, std::span<std::pair<int, Matrix> > data) {
    int accepted = 0;
    for(const auto &it : data) {
        auto ans = nn.forwardPropagate(it.second);

        int mx = 0;
        for(int i = 0; i < ans.M; i ++) { if(ans(0, i) > ans(0, mx)) { mx = i; } }
        accepted += mx == it.first;
    }
    return (double)accepted / (double)data.size() * 100.0;
}

void minibatchTraining(NeuralNetwork &nn, std::span<std::pair<int, Matrix> > data, const double alpha = 0.01) {
    auto getTotalCost = [&]() {
        double ret = 0;
        for(const auto &it : data) {
            Matrix ans = Matrix(1, LABEL_CNT);
            ans(0, it.first) = 1;
            ret += nn.cost(it.second, ans);
        }
        return ret / (double)data.size();
    };
    std::cout << "Minibatch started " << getTotalCost() << std::endl;

    std::pair<std::vector<Matrix>, std::vector<Matrix> > delta;
    for(size_t ind = 0; ind < nn.weight.size(); ind ++) {
        delta.first.push_back(Matrix(nn.weight[ind].N, nn.weight[ind].M));
        delta.second.push_back(Matrix(nn.bias[ind].M, nn.bias[ind].M));
    }

    for(const auto &it : data) {
        Matrix ans = Matrix(1, LABEL_CNT);
        ans(0, it.first) = 1;
        nn.backPropagate(it.second, ans, delta);
    }

    for(size_t ind = 0; ind < nn.weight.size(); ind ++) {
        for(size_t i = 0; i < nn.weight[ind].N; i ++) {
            for(size_t j = 0; j < nn.weight[ind].M; j ++) {
                nn.weight[ind](i, j) -= delta.first[ind](i, j) * alpha / (double)data.size();
            }
        }
        for(size_t i = 0; i < nn.bias[ind].M; i ++) {
            nn.bias[ind](0, i) -= delta.second[ind](0, i) * alpha / (double)data.size();
        }
    }
}


void fullTraining(NeuralNetwork &nn, std::vector<std::pair<int, Matrix> > &data) {
    std::span dataSpan = data;

    const size_t EPOCHS_CNT = 5;
    const size_t BATCH_CNT = 10;
    const size_t BATCH_LEN = data.size() / BATCH_CNT;
    const size_t BATCH_TIMES = 10;

    std::cout << "Starting training" << std::endl;
    for(size_t ind = 0; ind < EPOCHS_CNT; ind ++) {
        std::cout << "Shuffling " << ind << std::endl;
        std::random_shuffle(data.begin(), data.end());

        for(int i = BATCH_CNT - 1; i >= 0; i --) {
            for(int j = 0; j < BATCH_TIMES; j ++) {
                minibatchTraining(nn, dataSpan.subspan(BATCH_LEN * i, std::min(data.size(), BATCH_LEN * (i + 1)) - BATCH_LEN * i), 1);
            }
        }
        // TODO: This is not representative at all
        std::cout << "Epoch ended: " << getAcceptanceRate(nn, dataSpan.subspan(0, BATCH_LEN)) << std::endl;
    }
}


void saveToFile(const NeuralNetwork &nn, const std::string &fileName) {
    std::ofstream out(fileName);
    out << nn.layerLength.size() << std::endl;
    for(const auto &it : nn.layerLength) {
        out << it << " ";
    }
    out << std::endl;
    for(const auto &it : nn.weight) {
        for(size_t i = 0; i < it.N; i ++) {
            for(size_t j = 0; j < it.M; j ++) {
                out << it.get(i, j) << " ";
            }
            out << std::endl;
        }
    }
    for(const auto &it : nn.bias) {
        for(size_t i = 0; i < it.N; i ++) {
            for(size_t j = 0; j < it.M; j ++) {
                out << it.get(i, j) << " ";
            }
            out << std::endl;
        }
    }
    out.close();
}

std::vector<std::pair<int, Matrix> > readInput(const std::string &fileName) {
    std::fstream in(fileName);
    std::vector<std::pair<int, Matrix > > ret;

    std::string line;
    bool thrownAway = false;

    while(std::getline(in, line)) {
        if(!thrownAway) {thrownAway = true; continue;}

        for(size_t i = 0; i < line.size(); i ++) {
            if(line[i] == ',') {
                line[i] = ' ';
            }
        }
        std::stringstream str(line);

        int label;
        str >> label;

        Matrix image(1, PIC_N * PIC_M);
        for(int i = 0; i < PIC_N; i ++) {
            for(int j = 0; j < PIC_M; j ++) {
                double x;
                str >> x;

                image(0, i * PIC_M + j) = x / 255.;
            }
        }

        ret.push_back({label, image});
    }

    in.close();

    return ret;
}

std::vector<Matrix> readTest(const std::string &fileName) {
    std::fstream in(fileName);
    std::vector<Matrix > ret;

    std::string line;
    bool thrownAway = false;

    while(std::getline(in, line)) {
        if(!thrownAway) {thrownAway = true; continue;}

        for(size_t i = 0; i < line.size(); i ++) {
            if(line[i] == ',') {
                line[i] = ' ';
            }
        }
        std::stringstream str(line);

        Matrix image(1, PIC_N * PIC_M);
        for(int i = 0; i < PIC_N; i ++) {
            for(int j = 0; j < PIC_M; j ++) {
                double x;
                str >> x;

                image(0, i * PIC_M + j) = x / 255.;
            }
        }

        ret.push_back(image);
    }

    in.close();
    return ret;
}

void testNetwork(const NeuralNetwork &nn, const std::string &testFile, const std::string &outFile) {
    auto test = readTest(testFile);
    std::ofstream out(outFile);
    out << "ImageId,Label" << std::endl;
    for(int i = 0; i < test.size(); i ++) {
        auto ans = nn.forwardPropagate(test[i]);
        int mx = 0;
        for(int j = 0; j < LABEL_CNT; j ++) { if(ans(0, j) > ans(0, mx)) { mx = j; } }

        out << i + 1 << "," << mx << std::endl;
    }
    out.close();    
}

int main() {
    std::srand(time(NULL));

    auto data = readInput("train.csv");

    std::cout << "Read input - " << data.size() << " images loaded" << std::endl;

    NeuralNetwork nn({PIC_N * PIC_M, 50, 25, LABEL_CNT});
    fullTraining(nn, data);

    testNetwork(nn, "test.csv", "out.csv");

    saveToFile(nn, "nn.txt");

    return 0;
}
