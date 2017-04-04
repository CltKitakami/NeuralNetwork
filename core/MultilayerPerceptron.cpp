#include <algorithm>
#include "MultilayerPerceptron.hpp"
#include "common/Exception.hpp"


namespace nn {


MultilayerPerceptron::MultilayerPerceptron() :
	Network(),
	isInitialized(false),
	isTesting(false)
{
}


void MultilayerPerceptron::setHiddenNode(
	const std::vector<SizeType> &numberOfHiddenNode)
{
	this->numberOfHiddenNode = numberOfHiddenNode;
	isInitialized = false;
}


void MultilayerPerceptron::setActivation(
	const std::vector<Activation *> &activations)
{
	this->activations = activations;
	isInitialized = false;
}


MatrixType MultilayerPerceptron::getPrediction() const
{
	if (nodes.size() == 0)
		throw Exception(TRACE, "you do not train data yet");

	return nodes.back();
}


double MultilayerPerceptron::getAccuracy() const
{
	if (nodes.size() == 0)
		throw Exception(TRACE, "you do not train data yet");

	if (output.size() != nodes.back().size())
		throw Exception(TRACE, "you do not test data yet");

	MatrixType::Index groundTruth, prediction;
	size_t hitCount = 0;

	for (SizeType i = 0; i < output.rows(); ++i)
	{
		output.row(i).maxCoeff(&groundTruth);
		nodes.back().row(i).maxCoeff(&prediction);

		if (groundTruth == prediction)
			hitCount += 1;
	}

	return (double)hitCount / output.rows();
}


void MultilayerPerceptron::setOutput(MatrixType output)
{
	Network::setOutput(output);

	if (isInitialized == true)
		initializeDropoutMask();
}


void MultilayerPerceptron::train(
	const MatrixType &input, const MatrixType &output)
{
	Network::train(input, output);
}


void MultilayerPerceptron::test(
	const MatrixType &input, const MatrixType &output)
{
	Network::test(input, output);
}


void MultilayerPerceptron::train()
{
	if (isInitialized == false)
		initialize();

	if ((size_t)nodes[0].rows() != batchSize)
		initializeNodes(batchSize);

	for (size_t i = 0; i < epoch; ++i)
	{
		setBatch();
		forward();
		backward();
	}
}


void MultilayerPerceptron::test()
{
	if (isInitialized == false)
		throw Exception(TRACE, "you do not train data yet");

	if (nodes[0].rows() != input.rows())
		initializeNodes((size_t)input.rows());

	nodes[0].rightCols(input.cols()) = input;
	isTesting = true;

	forward();

	isTesting = false;
}


void MultilayerPerceptron::initialize()
{
	initializeWeights();
	initializeNodes(batchSize);
	initializeDropoutMask();
	checkVariables();

	isInitialized = true;
}


void MultilayerPerceptron::initializeWeights()
{
	SizeType numberOfInputNode = input.cols();
	SizeType numberOfOutputNode = output.cols();
	size_t numberOfHiddenLayer = numberOfHiddenNode.size();

	if (numberOfHiddenLayer == 0)
		throw Exception(TRACE, "numberOfHiddenLayer == 0");

	weights.clear();
	weights.reserve(numberOfHiddenLayer + 1);

	weights.push_back(MatrixType::Random(
		numberOfHiddenNode[0], numberOfInputNode + 1));

	for (size_t i = 1; i < numberOfHiddenLayer; ++i)
	{
		weights.push_back(MatrixType::Random(
			numberOfHiddenNode[i], numberOfHiddenNode[i - 1] + 1));
	}

	weights.push_back(MatrixType::Random(
		numberOfOutputNode, numberOfHiddenNode[numberOfHiddenLayer - 1] + 1));

	if (momentum != 0.0)
	{
		deltaWeights.clear();
		deltaWeights.reserve(weights.size());

		for (size_t i = 0; i < weights.size(); ++i)
		{
			deltaWeights.push_back(MatrixType::Zero(
				weights[i].rows(), weights[i].cols()));
		}
	}
}


void MultilayerPerceptron::initializeNodes(size_t numberOfSample)
{
	SizeType numberOfInputNode = input.cols();
	SizeType numberOfOutputNode = output.cols();
	size_t numberOfActivation = activations.size();

	if (numberOfActivation == 0)
		throw Exception(TRACE, "numberOfActivation == 0");

	nodes.clear();
	nodes.reserve(numberOfActivation + 1);

	nodes.push_back(MatrixType(numberOfSample, numberOfInputNode + 1));

	for (size_t i = 1; i < numberOfActivation; ++i)
	{
		nodes.push_back(
			MatrixType(numberOfSample, numberOfHiddenNode[i - 1] + 1));
	}

	nodes.push_back(MatrixType(numberOfSample, numberOfOutputNode));

	for (size_t i = 0; i < nodes.size() - 1; ++i)
		nodes[i].col(0).setOnes();	// set bias term except output layer
}


void MultilayerPerceptron::initializeDropoutMask()
{
	if (dropout > 0.0)
	{
		dropoutMask.clear();
		dropoutMask.reserve(nodes.size() - 2);

		for (size_t i = 1; i < nodes.size() - 1; ++i)
		{
			dropoutMask.push_back(MatrixType(
				nodes[i].rows(), nodes[i].cols() - 1));
		}
	}
}


void MultilayerPerceptron::checkVariables()
{
	if (weights.size() != activations.size())
	{
		throw Exception(TRACE,
			"you must set right hidden and output layer activations");
	}

	if (input.rows() != output.rows())
		throw Exception(TRACE, "number of samples do not match");

	if (loss == nullptr)
		throw Exception(TRACE, "you must assign loss function");
}


void MultilayerPerceptron::shuffleSamples()
{
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
		perm(input.rows());

	perm.setIdentity();
	std::random_shuffle(perm.indices().data(),
		perm.indices().data() + perm.indices().size());

	input = perm * input;
	output = perm * output;
}


void MultilayerPerceptron::setBatch()
{
	if (batchOffset + batchSize > (size_t)input.rows())
		batchOffset = 0;

	if (batchOffset == 0)
		shuffleSamples();

	nodes[0].rightCols(input.cols()) = input.block(
		(SizeType)batchOffset, 0, (SizeType)batchSize, input.cols());

	batchOutput = output.block(
		(SizeType)batchOffset, 0, (SizeType)batchSize, input.cols());

	batchOffset += batchSize;
}


void MultilayerPerceptron::forward()
{
	for (size_t i = 1; i < nodes.size(); ++i)
	{
		MatrixType &&m = nodes[i - 1] * weights[i - 1].transpose();

		activations[i - 1]->forward(m);

		if (i != nodes.size() - 1)	// hidden layer output nodes
		{
			nodes[i].rightCols(nodes[i].cols() - 1).noalias() =
				dropout > 0.0 ? applyDroupout(m, i - 1) : m;
		}
		else	// output layer output nodes
		{
			nodes.back().noalias() = m;
		}
	}
}


void MultilayerPerceptron::backward()
{
	MatrixType &&error = loss->compute(batchOutput, nodes.back());

	for (size_t i = nodes.size() - 1; i > 0; --i)
	{
		MatrixType dW;

		activations[i - 1]->backward(nodes[i]);

		if (i == nodes.size() - 1)	// output layer first derivatives
		{
			nodes.back().noalias() = nodes.back().cwiseProduct(error);
			dW.noalias() = nodes.back().transpose() * nodes[nodes.size() - 2];
		}
		else	// hidden layer first derivatives
		{
			nodes[i].noalias() = nodes[i].cwiseProduct(
				nodes[i + 1] * weights[i]);

			// use layer nodes withot bias term
			auto &&m = nodes[i].rightCols(nodes[i].cols() - 1);

			if (dropout > 0.0)
			{
				dW.noalias() = m.cwiseProduct(
					dropoutMask[i - 1]).transpose() * nodes[i - 1];
			}
			else
			{
				dW.noalias() = m.transpose() * nodes[i - 1];
			}
		}

		updateWeight(dW, i - 1);
	}
}


MatrixType MultilayerPerceptron::applyDroupout(MatrixType &m, size_t layerIndex)
{
	if (isTesting == false)
	{
		MatrixType random = (MatrixType::Random(m.rows(), m.cols())
			+ MatrixType::Ones(m.rows(), m.cols())) / 2.0;

		dropoutMask[layerIndex] = (random.array() > dropout).select(
			MatrixType::Ones(m.rows(), m.cols()),
			MatrixType::Zero(m.rows(), m.cols()));

		return m.cwiseProduct(dropoutMask[layerIndex]);
	}
	else
	{
		return m * (1.0 - dropout);
	}
}


void MultilayerPerceptron::updateWeight(MatrixType &dW, size_t layerIndex)
{
	if (momentum == 0.0)
	{
		weights[layerIndex] += dW * learningRate / batchSize;
	}
	else
	{
		deltaWeights[layerIndex] *= momentum;
		deltaWeights[layerIndex] += dW * learningRate / batchSize;
		weights[layerIndex] += deltaWeights[layerIndex];
	}
}


}
