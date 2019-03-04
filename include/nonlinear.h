#include <math.h>
#ifndef NONLINEAR_H
#define NONLINEAR_H

//SIEKNET KERNEL START

#define SIGMOID(x)    (1/(1+exp(-x)))
#define HYPERTAN(x)   ((exp(x) - exp(-x))/exp(x) + exp(-x))
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

#ifdef SIEKNET_BUILD_KERNEL //gets inserted before clBuildProgram is run

__kernel void linear_kernel(__global float *x, __global float *z, __global float *params, int dim, int layer_param_idx){
	const int i = get_global_id(0);
	z[i] = 0;
	const int w_idx = layer_param_idx + ((dim + 1) * i);
	float sum = 0;
	for(int j = 0; j < dim; j++){
		sum += x[j] * params[w_idx + j + 1]; //weights
	}
	z[i] = sum + params[w_idx]; //wx + b
}

__kernel void sigmoid_kernel(__global float *x, __global float *y){
	const int i = get_global_id(0);
	y[i] = activate(x[i], sigmoid);
}

__kernel void propagate_grads(__global float *grads, __global float *output, __global float *dest, __global float *params, __global float *param_grads, Nonlinearity nonlinearity_type, int layer_param_idx, int neurons, int dim){
	const int i = get_global_id(0);

	dest[i] = 0;
	for(int j = 0; j < neurons; j++){
		const int w_idx = layer_param_idx + ((dim + 1) * j) + i;
		float w = params[w_idx];
		float d = differentiate(output[j], nonlinearity_type);
		float g = grads[j];
		dest[i] += w * d * g;

		param_grads[w_idx] += 
	}
	for(int j = 0; j < neurons; j++){

	}
}

#endif

//SIEKNET KERNEL END

#endif
