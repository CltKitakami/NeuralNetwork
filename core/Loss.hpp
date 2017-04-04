#ifndef _LOSS_HPP_
#define _LOSS_HPP_

#include "nn.hpp"


namespace nn {


class Loss
{
public:
	Loss() = default;
	virtual ~Loss() = default;
	virtual MatrixType compute(const MatrixType &, const MatrixType &) = 0;
};


}

#endif
