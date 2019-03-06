#include "optimizer.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef GPU
static float cpu_sgd_step(SGD o){
	float entropy = 0;
	for(int i = 0; i < o.num_params; i++){
		o.weights[i] += o.learning_rate * o.gradient[i];
		o.gradient[i] = 0.0;
	}
	return entropy;
}

static float cpu_momentum_step(Momentum o){
	float entropy = 0;
	for(int i = 0; i < o.num_params; i++){
      o.z[i] = o.beta * o.z[i] + o.gradient[i];
      o.weights[i] += o.alpha * o.z[i];
      o.gradient[i] = 0.0;
	}
	return entropy;
}

SGD cpu_init_SGD(float *weights, float *gradient, size_t num_params){
	SGD o;
	o.weights = weights;
	o.gradient = gradient;
	o.num_params = num_params;
	o.learning_rate = 0.05;
	o.step = cpu_sgd_step;
	return o;
}

Momentum cpu_init_Momentum(float *weights, float *gradient, size_t num_params){
	Momentum o;
	o.weights = weights;
	o.gradient = gradient;

	o.z = (float*)malloc(num_params*sizeof(float));
	memset(o.z, '\0', num_params*sizeof(float));

	o.num_params = num_params;
	o.alpha = 0.001;
	o.beta = 0.99;
	o.step = cpu_momentum_step;
	return o;
}
#else
static float gpu_sgd_step(SGD o){
	float entropy = 0;
	return entropy;
}

static float gpu_momentum_step(Momentum o){
	float entropy = 0;
	return entropy;
}

SGD gpu_init_SGD(cl_mem weights, cl_mem gradient, size_t num_params, cl_context c, cl_command_queue q){
	SGD o;
	o.weights = weights;
	o.gradient = gradient;
	o.num_params = num_params;
	o.learning_rate = 0.05;
	o.step = gpu_sgd_step;
	return o;
}

Momentum gpu_init_Momentum(cl_mem weights, cl_mem gradient, size_t num_params, cl_context c, cl_command_queue q){
	Momentum o;
	o.weights = weights;
	o.gradient = gradient;

	o.z = (float*)malloc(num_params*sizeof(float));
	memset(o.z, '\0', num_params*sizeof(float));

	o.num_params = num_params;
	o.alpha = 0.001;
	o.beta = 0.99;
	o.step = gpu_momentum_step;
	return o;
}
#endif
