#ifndef NONLINEAR_H
#include <math.h>

#define SIGMOID(x)    (1/(1+exp(-x)))
#define TANH(x)       ((exp(x) - exp(-x))/exp(x) + exp(-x))
#define SOFTMAX(x, y) (exp(x)/y)
#define RELU(x)       ((0 <= x) * x)

#define D_SIGMOID(x)  (x*(1-x))
#define D_TANH(x)     (1 - x*x)
#define D_SOFTMAX(x)  (x*(1-x))
#define D_RELU(x)     ((0 <= x) * 1)


#endif
