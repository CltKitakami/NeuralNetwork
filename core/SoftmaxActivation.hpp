#ifndef _SOFTMAXACTIVATION_HPP_
#define _SOFTMAXACTIVATION_HPP_

#include <cmath>
#include "Activation.hpp"


namespace nn {

class SoftmaxActivation : public Activation

{
public:
	virtual ~SoftmaxActivation() = default;

	virtual void forward(MatrixType &x) override
	{
		auto &&m = x.colwise() - x.rowwise().maxCoeff();
		auto &&e = m.unaryExpr([](double v){ return std::exp(v); });
		auto &&sum = e.rowwise().sum().replicate(1, x.cols());
		x.noalias() = e.cwiseQuotient(sum);
	}

	virtual void backward(MatrixType &x) override
	{
		x = MatrixType::Ones(x.rows(), x.cols());
	};
};


}

#endif
