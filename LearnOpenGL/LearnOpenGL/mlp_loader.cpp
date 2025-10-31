#include "mlp_loader.h"

#include <fstream>
#include <sstream>
#include <limits>

namespace
{
bool ReadFloatLine(std::ifstream& file, std::vector<float>& values, std::size_t expectedCount)
{
    values.clear();
    std::string line;
    if (!std::getline(file, line))
    {
        return false;
    }
    std::stringstream ss(line);
    float value = 0.0f;
    while (ss >> value)
    {
        values.push_back(value);
    }
    return values.size() == expectedCount;
}
}

bool LoadMlpWeights(const std::string& path, MlpNetwork& network, std::string& errorMessage)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        errorMessage = "Failed to open MLP weight file: " + path;
        return false;
    }

    std::string header;
    std::getline(file, header);

    file >> network.inputDim >> network.hiddenDim >> network.hiddenDim2 >> network.outputDim;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<float> buffer;
    if (!ReadFloatLine(file, buffer, static_cast<std::size_t>(network.inputDim)))
    {
        errorMessage = "Failed to read input mean";
        return false;
    }
    network.inputMean = buffer;

    if (!ReadFloatLine(file, buffer, static_cast<std::size_t>(network.inputDim)))
    {
        errorMessage = "Failed to read input standard deviation";
        return false;
    }
    network.inputStd = buffer;

    auto readLayer = [&](int inDim, int outDim, std::vector<float>& weights, std::vector<float>& bias, const char* layerName) -> bool
    {
        if (!ReadFloatLine(file, buffer, static_cast<std::size_t>(inDim * outDim)))
        {
            errorMessage = std::string("Failed to read weights for ") + layerName;
            return false;
        }
        weights = buffer;

        if (!ReadFloatLine(file, buffer, static_cast<std::size_t>(outDim)))
        {
            errorMessage = std::string("Failed to read bias for ") + layerName;
            return false;
        }
        bias = buffer;
        return true;
    };

    if (!readLayer(network.inputDim, network.hiddenDim, network.layer0Weights, network.layer0Bias, "layer 0"))
    {
        return false;
    }
    if (!readLayer(network.hiddenDim, network.hiddenDim2, network.layer1Weights, network.layer1Bias, "layer 1"))
    {
        return false;
    }
    if (!readLayer(network.hiddenDim2, network.outputDim, network.layer2Weights, network.layer2Bias, "layer 2"))
    {
        return false;
    }

    return true;
}
