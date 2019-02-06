#ifndef OPTIM_H
#define OPTIM_H

#include <stdlib.h>
#include <stdio.h>

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

SGD init_sgd(float *, float *, size_t);
Momentum init_momentum(float *, float *, size_t);

#endif
