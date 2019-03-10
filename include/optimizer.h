#ifndef OPTIM_H
#define OPTIM_H

#include "conf.h"
#ifdef GPU
#include "opencl_utils.h"
#endif

#include <stdlib.h>
#include <stdio.h>

#ifdef GPU
#define create_optimizer(TYPE, network) gpu_init_## TYPE ((network).gpu_params, (network).param_grad, (network).num_params)
#else
#define create_optimizer(TYPE, network) cpu_init_## TYPE ((network).params, (network).param_grad, (network).num_params)
#endif

typedef struct sgd{
#ifdef GPU
	cl_mem weights;
	cl_mem gradient;
	cl_command_queue q;
	cl_context c;
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
	cl_command_queue q;
	cl_context c;
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
SGD gpu_init_SGD(cl_mem, cl_mem, size_t);
Momentum gpu_init_Momentum(cl_mem, cl_mem, size_t);
#else
SGD cpu_init_SGD(float *, float *, size_t);
Momentum cpu_init_Momentum(float *, float *, size_t);
#endif
#endif
