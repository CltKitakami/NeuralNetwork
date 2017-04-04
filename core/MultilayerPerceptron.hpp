#ifndef _MULTILAYERPERCEPTRON_HPP_
#define _MULTILAYERPERCEPTRON_HPP_

#include <vector>
#include "Network.hpp"
#include "Activation.hpp"


namespace nn {


class MultilayerPerceptron : public Network
{
public:
	MultilayerPerceptron();
	virtual ~MultilayerPerceptron() = default;

	void setHiddenNode(const std::vector<SizeType> &numberOfHiddenNode);
	std::vector<SizeType> getHiddenNode() const { return numberOfHiddenNode; }

	void setActivation(const std::vector<Activation *> &activations);
	std::vector<Activation *> getActivation() const { return activations; }

	MatrixType getPrediction() const;
	double getAccuracy() const;

	virtual void setOutput(MatrixType output) override;

	virtual	void train(
		const MatrixType &input, const MatrixType &output) override;
	virtual	void train() override;

	virtual	void test(
		const MatrixType &input, const MatrixType &output) override;
	virtual	void test() override;

private:
	void initialize();
	void initializeWeights();
	void initializeNodes(size_t numberOfSample);
	void initializeDropoutMask();
	void checkVariables();
	void shuffleSamples();
	void setBatch();
	void forward();
	void backward();
	MatrixType applyDroupout(MatrixType &m, size_t layerIndex);
	void updateWeight(MatrixType &dW, size_t layerIndex);

	std::vector<MatrixType> weights;
	std::vector<MatrixType> deltaWeights;
	std::vector<MatrixType> dropoutMask;
	std::vector<MatrixType> nodes;
	std::vector<Activation *> activations;
	std::vector<SizeType> numberOfHiddenNode;
	MatrixType batchOutput;
	bool isInitialized;
	bool isTesting;
};


}

#endif
