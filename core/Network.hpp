#ifndef _NETWORK_HPP_
#define _NETWORK_HPP_

#include "nn.hpp"
#include "Loss.hpp"


namespace nn {


class Network
{
public:
	Network();
	virtual ~Network() = default;

	virtual void setInput(MatrixType input) { this->input = input; }
	virtual MatrixType getInput() const { return input; }

	virtual void setOutput(MatrixType output) { this->output = output; }
	virtual MatrixType getOutput() const { return output; }

	virtual void setLearningRate(ValueType rate) { learningRate = rate; }
	virtual ValueType getLearningRate() const { return learningRate; }

	virtual void setMomentum(ValueType momentum) { this->momentum = momentum; }
	virtual ValueType getMomentum() const { return momentum; }

	virtual void setDropout(ValueType dropout);
	virtual ValueType getDropout() const { return dropout; }

	virtual void setBatchSize(size_t size) { batchSize = size; }
	virtual size_t getBatchSize() const { return batchSize; }

	virtual void setEpoch(size_t epoch) { this->epoch = epoch; }
	virtual size_t getEpoch() const { return epoch; }

	virtual void setLoss(Loss *loss) { this->loss = loss; }
	virtual Loss * getLoss() const { return loss; }

	virtual	void train(const MatrixType &input, const MatrixType &output);
	virtual	void train() = 0;

	virtual	void test(const MatrixType &input, const MatrixType &output);
	virtual	void test() = 0;

protected:
	MatrixType input, output;
	ValueType learningRate;
	ValueType momentum;
	ValueType dropout;
	size_t batchSize;
	size_t epoch;
	size_t batchOffset;
	Loss *loss;
};


}

#endif
