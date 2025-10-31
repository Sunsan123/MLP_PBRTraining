#pragma once

#include <string>
#include <vector>

struct MlpNetwork
{
    int inputDim = 0;
    int hiddenDim = 0;
    int hiddenDim2 = 0;
    int outputDim = 0;
    std::vector<float> inputMean;
    std::vector<float> inputStd;
    std::vector<float> layer0Weights;
    std::vector<float> layer0Bias;
    std::vector<float> layer1Weights;
    std::vector<float> layer1Bias;
    std::vector<float> layer2Weights;
    std::vector<float> layer2Bias;
};

bool LoadMlpWeights(const std::string& path, MlpNetwork& network, std::string& errorMessage);
