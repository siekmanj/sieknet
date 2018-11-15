#ifndef GA_H
#define GA_H
#include "MLP.h"

typedef struct population {
	MLP *pool;
	size_t size;
	float mutation_rate;
	float plasticity;

} Pool;

void pool_from_mlp(MLP *, Pool *);
void evolve(Pool *);
void evolve_safe(Pool *);
//void safe_mutate(Pool *);
//void mutate(MLP *);
void dealloc_pool(Pool *);

//MLP crossbreed(MLP *, MLP *);
//MLP copy(MLP *);

#endif
