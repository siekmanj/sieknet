#ifndef OPTIM_H
#define OPTIM_H

#include "opencl_utils.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef GPU
#define create_optimizer(TYPE, network) init_## TYPE ((network).gpu_params, (network).param_grad, (network).num_params)
#else
#define create_optimizer(TYPE, network) init_## TYPE ((network).params, (network).param_grad, (network).num_params)
#endif

typedef struct sgd{
#ifdef GPU
	cl_mem weights;
	cl_mem gradients;
#else
	float *weights;
	float *gradient;
#endif
	float learning_rate;
	size_t num_params;
	float (*step)(struct sgd);

} SGD;

typedef struct momentum{
#ifdef GPU
	cl_mem weights;
	cl_mem gradient;
	cl_mem z;
#else
	float *weights;
	float *gradient;
	float *z;
#endif
	float alpha;
	float beta;
	size_t num_params;
	float (*step)(struct momentum);
} Momentum;

typedef struct adam{
	/*...*/
} Adam;

#ifdef GPU
SGD init_SGD(cl_mem, cl_mem, size_t, cl_context c, cl_queue q);
Momentum init_Momentum(cl_mem, cl_mem, size_t, cl_context c, cl_queue c);
#else
SGD init_SGD(float *, float *, size_t);
Momentum init_Momentum(float *, float *, size_t);
#endif

#endif
