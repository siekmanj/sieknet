#ifndef RANDOM_SEARCH_H
#define RANDOM_SEARCH_H

#include <stdlib.h>
#include <stdio.h>
#include <optimizer.h>

typedef enum search_type_t {BASIC, V1, V1_t, V2, V2_t} Search_type;

typedef struct delta_{
	float *p;
	float r_pos;
	float r_neg;
} Delta;

typedef struct ars_{
	float std;
	float step_size;
	float top_b;

	size_t directions;
	size_t num_params;
  size_t num_threads;

	float *params;
	float *update;
	Delta **deltas;

  float (*f)(const float *, size_t, Normalizer*);

  Normalizer *normalizer;
	Search_type algo;
	SGD optim;
} ARS;

ARS create_ars(float (*R)(const float *, size_t, Normalizer*), float *, size_t, size_t);

void ars_step(ARS);

#endif
