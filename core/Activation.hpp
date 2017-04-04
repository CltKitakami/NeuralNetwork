#ifndef _ACTIVATION_HPP_
#define _ACTIVATION_HPP_

#include "nn.hpp"


namespace nn {


class Activation
{
public:
	Activation() = default;
	virtual ~Activation() = default;

	virtual void forward(MatrixType &) = 0;
	virtual void backward(MatrixType &) = 0;
};


}

#endif
