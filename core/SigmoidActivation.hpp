#ifndef _SIGMOIDACTIVATION_HPP_
#define _SIGMOIDACTIVATION_HPP_

#include "Activation.hpp"


namespace nn {


class SigmoidActivation : public Activation
{
public:
	virtual ~SigmoidActivation() = default;

	virtual void forward(MatrixType &x) override
	{
		ValueType *p = x.data();

		for (int i = 0; i < x.size(); ++i)
			p[i] = ValueType(1.0 / (1.0 + std::exp(-p[i])));
	}

	virtual void backward(MatrixType &x) override
	{
		x.noalias() = x.cwiseProduct(
			MatrixType::Ones(x.rows(), x.cols()) - x);
	}
};


}

#endif
