#ifndef _LINEARACTIVATION_HPP_
#define _LINEARACTIVATION_HPP_

#include "Activation.hpp"


namespace nn {


class LinearActivation : public Activation
{
public:
	virtual ~LinearActivation() = default;
	virtual void forward(MatrixType &) override {}
	virtual void backward(MatrixType &x) override
	{
		x = MatrixType::Ones(x.rows(), x.cols());
	};
};


}

#endif
