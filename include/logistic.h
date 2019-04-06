/* This file is used by both the CPU and GPU implementations. */

#include <math.h>
#ifndef NONLINEAR_H
#define NONLINEAR_H

/*<<KERNEL START>>*/


typedef enum nonlin{
	sigmoid,
	hypertan,
	relu,
	softmax
} Nonlinearity;

typedef enum costfn{
	cross_entropy,
	quadratic
} Costfn;

static float SIGMOID(float x){
	return 1/(1 + exp(-x));
}

static float HYPERTAN(float x){
	if(x > 7.0f)  return 0.999998f;
	if(x < -7.0f) return -0.999998f;
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

#define RELU(x)       ((0 <= x) * x)


#define D_SIGMOID(x)  (x*(1-x))
#define D_HYPERTAN(x) (1 - x*x)
#define D_SOFTMAX(x)  (x*(1-x))
#define D_RELU(x)     ((0 <= x) * 1)


static float CROSS_ENTROPY(float o, float y){
	float o_n = o;
	if(o_n > 0.9999f) o_n = 0.9999f;
	if(o_n < 0.0001f) o_n = 0.0001f;
	return (-(y * log(o_n) + (1-y) * log(1-o_n)));
}

static float QUADRATIC(float o, float y){
	return (0.5f*(y-o) * (y-o));
}

static float D_CROSS_ENTROPY(float o, float y){
	float o_n = o;
	if(o_n > 0.9999f) o_n = 0.9999f;
	if(o_n < 0.0001f) o_n = 0.0001f;
	return (y/o_n) - ((1-y)/(1-o_n));
}

static float D_QUADRATIC(float o, float y){
	return y - o;
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
	return 0.0f;
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
		case softmax:
			//do nothing
			break;
	}
	return 0.0f;
}

static float cost(float o, float y, Costfn c){
	switch(c){
		case cross_entropy:
			return CROSS_ENTROPY(o, y);
			break;
		case quadratic:
			return QUADRATIC(o, y);
			break;
		//case default:
		//	return -1.0f;
		//	break;
	}
	return -1.0f;
}

static float cost_gradient(float o, float y, Costfn c){
	switch(c){
		case cross_entropy:
			return D_CROSS_ENTROPY(o, y);
			break;
		case quadratic:
			return D_QUADRATIC(o, y);
			break;
		//case default:
			//return 0.0f;
		//	break;
	}
	return 0.0f;
}
static void no_op(){}

#define agnostic_softmax_kernel(z, y, sum, i) \
	y[i] = exp(z[i]) / (*sum)

#define agnostic_softmax_sum_kernel(z, sum, dim) \
	*sum = 0.0f;                   \
	float fmax = 0.0f;             \
	for(int i = 0; i < dim; i++)   \
		if(z[i] > fmax) fmax = z[i]; \
	for(int i = 0; i < dim; i++){  \
		*sum = *sum + exp(z[i]-fmax);      \
		z[i] = z[i] - fmax;          \
	}                              \
	no_op()


#define agnostic_cost_kernel(o, y, c, dim, type) \
	*c = 0.0f;                      \
	for(int i = 0; i < dim; i++){     \
		*c = *c + cost(o[i], y[i], type); \
	}                                 \
	no_op()

#define agnostic_cost_gradient_kernel(o, y, dest, type, i) \
	{                                            \
		dest[i] = cost_gradient(o[i], y[i], type); \
	}                                            \
	no_op()
/*<<KERNEL END>>*/

#endif
