#include "Network.hpp"
#include "common/Exception.hpp"


namespace nn {


Network::Network() :
	learningRate(0.001),
	momentum(0.0),
	dropout(0.0),
	batchSize(1),
	epoch(1),
	batchOffset(0),
	loss(nullptr)
{
}


void Network::setDropout(ValueType dropout)
{
	if (0.0 > dropout || dropout > 1.0)
		throw Exception(TRACE, "dropout interval is 0.0-1.0");

	this->dropout = dropout;
}


void Network::train(const MatrixType &input, const MatrixType &output)
{
	this->input = input;
	this->output = output;

	train();
}


void Network::test(const MatrixType &input, const MatrixType &output)
{
	this->input = input;
	this->output = output;

	test();
}


}
