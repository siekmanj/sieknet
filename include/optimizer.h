#ifndef OPTIM_H
#define OPTIM_H

#include <stdlib.h>
#include <stdio.h>

#define create_optimizer(TYPE, network) init_## TYPE ((network).params, (network).param_grad, (network).num_params)

typedef struct sgd{
	float *weights;
	float *gradient;
	float learning_rate;
	size_t num_params;
	float (*step)(struct sgd);

} SGD;

typedef struct momentum{
	float *weights;
	float *gradient;
	float *z;
	float alpha;
	float beta;
	size_t num_params;
	float (*step)(struct momentum);
} Momentum;

typedef struct adam{
	/*...*/
} Adam;

SGD init_SGD(float *, float *, size_t);
Momentum init_Momentum(float *, float *, size_t);

#endif
