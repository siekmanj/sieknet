#include "optimizer.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>


static float sgd_step(SGD o){
	float entropy = 0;
	for(int i = 0; i < o.num_params; i++){
		o.weights[i] += o.learning_rate * o.gradient[i];
		//entropy += o.learning_rate * o.gradient[i];
		o.gradient[i] = 0.0;
	}
	return entropy;
}

static float momentum_step(Momentum o){
	float entropy = 0;
	for(int i = 0; i < o.num_params; i++){
		o.z[i] = o.beta * o.z[i] + o.gradient[i];
		o.weights[i] += o.alpha * o.z[i];
		//entropy += o.alpha * o.z[i];
		o.gradient[i] = 0.0;
	}
	return entropy;
}

SGD init_sgd(float *weights, float *gradient, size_t num_params){
	SGD o;
	o.weights = weights;
	o.gradient = gradient;
	o.num_params = num_params;
	o.learning_rate = 0.05;
	o.step = sgd_step;
	return o;
}

Momentum init_momentum(float *weights, float *gradient, size_t num_params){
	Momentum o;
	o.weights = weights;
	o.gradient = gradient;

	o.z = (float*)malloc(num_params*sizeof(float));
	memset(o.z, '\0', num_params*sizeof(float));

	o.num_params = num_params;
	o.alpha = 0.001;
	o.beta = 0.99;
	o.step = momentum_step;
	return o;
}

