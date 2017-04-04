#ifndef _NN_HPP_
#define _NN_HPP_

#include <cstdint>
#include <eigen3/Eigen/Dense>


namespace nn {


typedef double ValueType;
typedef ssize_t SizeType;
typedef Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic> MatrixType;


}


#include "Network.hpp"
#include "Activation.hpp"
#include "LinearActivation.hpp"
#include "SigmoidActivation.hpp"
#include "SoftmaxActivation.hpp"
#include "Loss.hpp"


#endif
