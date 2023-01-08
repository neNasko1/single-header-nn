#include "src/my_math.h"
#include "src/small_nn.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>

const uint32_t IMAGE_N = 28;
const uint32_t IMAGE_M = 28;
const uint32_t LABEL_CNT = 10;

std::vector<std::pair<LinAlg::Matrix<1, IMAGE_N * IMAGE_M>, LinAlg::Matrix<1, LABEL_CNT>>> readInput(const std::string &fileName, const bool isTest = false) {
    std::fstream in(fileName);
    std::vector<std::pair<LinAlg::Matrix<1, IMAGE_N * IMAGE_M>, LinAlg::Matrix<1, LABEL_CNT>>> ret;

    std::string line;
    bool thrownAway = false;

    while(std::getline(in, line)) {
        if(!thrownAway) {thrownAway = true; continue;}

        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream str(line);

        LinAlg::Matrix<1, LABEL_CNT> labelMat(0);
        if(!isTest) {
            int label; str >> label;
            labelMat[0][label] = 1;
        }

        LinAlg::Matrix<1, IMAGE_N * IMAGE_M> image;
        for(size_t i = 0; i < IMAGE_N * IMAGE_M; i ++) {
            double x; str >> x;
            image[0][i] = x / 255.;
        }

        ret.push_back({image, labelMat});
    }

    return ret;
}

void outputNetworkResults(const auto nn, const auto testPoints, const std::string outfile) {
    std::ofstream out(outfile);
    out << "ImageId,Label" << std::endl;
    for(size_t i = 0; i < testPoints.size(); i ++) {
        const auto ans = nn.forwardPropagate(testPoints[i].first);
        const size_t mx = std::max_element(ans[0], ans[0] + LABEL_CNT) - ans[0];
        out << i + 1 << "," << mx << std::endl;
    }
}

int main() {
    #define FullyConnectedApplicationLayer(N, M, F) MachineLearning::DenseLayer<N, M>, MachineLearning::BiasLayer<M>, MachineLearning::ApplicationLayer<M, F>
    MachineLearning::NeuralNetwork<
        FullyConnectedApplicationLayer(IMAGE_N * IMAGE_M, 128, Continuous::Sigmoid),
        FullyConnectedApplicationLayer(128, 64, Continuous::Sigmoid),
        FullyConnectedApplicationLayer(64, LABEL_CNT, Continuous::Sigmoid)
    > nn;

    auto dataPoints = readInput("train.csv");
    std::cout << "Read data points " << dataPoints.size() << std::endl;
    MachineLearning::Trainer<Continuous::QuadraticLoss, decltype(nn)> trainer(&nn, dataPoints);
    trainer.run(2, 10, 420, 1);

    const auto testPoints = readInput("test.csv", true);
    outputNetworkResults(nn, testPoints, "output.csv");
}
