#include <math.h>
#ifndef NONLINEAR_H
#define NONLINEAR_H

/*<<KERNEL START>>*/

//#define SIGMOID(x)    (1/(1+exp(-x)))
//#define HYPERTAN(x)   ((x < 7.0 && x > -7.0) ? ((exp(x) - exp(-x))/(exp(x) + exp(-x))) : (x >= 7.0) ? 0.999998 : -0.999998)
#define SOFTMAX(x, y) (exp(x)/y)
#define RELU(x)       ((0 <= x) * x)

#define D_SIGMOID(x)  (x*(1-x))
#define D_HYPERTAN(x) (1 - x*x)
#define D_SOFTMAX(x)  (x*(1-x))
#define D_RELU(x)     ((0 <= x) * 1)

typedef enum nonlin{
	sigmoid,
	hypertan,
	relu,
	softmax
} Nonlinearity;

static float SIGMOID(float x){
	return 1/(1 + exp(-x));
}

static float HYPERTAN(float x){
	if(x > 7.0)  return 0.999998;
	if(x < -7.0) return -0.999998;
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

static float differentiate(float x, Nonlinearity n){
	switch(n){
		case sigmoid:
			return D_SIGMOID(x);
			break;
		case hypertan:
			return D_HYPERTAN(x);
			break;
		case relu:
			return D_RELU(x);
			break;
		case softmax:
			return D_SOFTMAX(x);
			break;
	}
	return 0;
}

static float activate(float x, Nonlinearity n){
	switch(n){
		case sigmoid:
			return SIGMOID(x);
			break;
		case hypertan:
			return HYPERTAN(x);
			break;
		case relu:
			return RELU(x);
			break;
		//case softmax:
		//	return SOFTMAX(x);
		//	break;
	}
	return 0;
}
/*<<KERNEL END>>*/


#endif
