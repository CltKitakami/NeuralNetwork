#include <iostream>
#include "core/nn.hpp"
#include "core/MultilayerPerceptron.hpp"


using namespace nn;


class SquaredErrorLoss : public Loss
{
	enum { DISPLAY_STEP = 100 };
	size_t displayCount;

public:
	SquaredErrorLoss() : displayCount(0) {}

	virtual ~SquaredErrorLoss() = default;

	virtual MatrixType compute(
		const MatrixType &groundTruth, const MatrixType &prediction)
	{
		MatrixType error = groundTruth - prediction;

		if (++displayCount == DISPLAY_STEP)
		{
			auto sqare = error.array().square();
			displayCount = 0;

			std::cout << "mean squared error: "
				<< sqare.sum() / prediction.rows() / 2 << std::endl;
		}

		return error;
	}
};


void printMatrix(const MatrixType &m, const std::string &name)
{
	std::cout << name << "(" << m.rows() <<
		", " << m.cols() << ") =\n" << m.format(2) << "\n";
}


void generateData(MatrixType &data, MatrixType &label, SizeType samples)
{
	MatrixType twoClass = MatrixType(2, 2);
	MatrixType x = (MatrixType::Random(samples, 2)
		+ MatrixType::Ones(samples, 2)) / 2.0;

	twoClass << 1, 0,
				0, 1;

	MatrixType y = twoClass.replicate(samples / 2, 1);

	data << x - y * 3;
	label << y;
}


int main()
{
	MultilayerPerceptron mlp;
	SigmoidActivation sigmoid;
	SoftmaxActivation softmax;
	SquaredErrorLoss loss;

	mlp.setLearningRate(0.01);
	mlp.setMomentum(0.1);
	mlp.setDropout(0.001);
	mlp.setBatchSize(10);
	mlp.setEpoch(1000);

	// two hidden layers, 10 nodes for each layer
	mlp.setHiddenNode({ 10, 10 });

	// two sigmoid activation functions for hidden layer
	// and softmax for output
	mlp.setActivation({ &sigmoid, &sigmoid, &softmax });

	mlp.setLoss(&loss);

	{
		const SizeType samples = 200;
		MatrixType data(samples, 2), label(samples, 2);

		generateData(data, label, samples);
		mlp.train(data, label);
	}

	{
		const SizeType samples = 20;
		MatrixType data(samples, 2), label(samples, 2);

		generateData(data, label, samples);
		mlp.test(data, label);

		printMatrix(mlp.getPrediction(), "prediction");
		printMatrix(data, "data");
		printMatrix(label, "label");

		std::cout << "accracy = " << mlp.getAccuracy() << std::endl;
	}

	return 0;
}
