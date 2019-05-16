#ifndef RANDOM_SEARCH_H
#define RANDOM_SEARCH_H

#include <stdlib.h>
#include <stdio.h>
#include <optimizer.h>

typedef enum search_type_t {BASIC, V1, V1_T, V2, V2_T} Search_type;

typedef struct delta_{
	float *p;
	float r_pos;
	float r_neg;
} Delta;

typedef struct RS{
	float std;
	float step_size;
	float cutoff;

	size_t directions;
	size_t num_params;

	float *params;
	float *update;
	Delta **deltas;

	Search_type algo;
	SGD optim;
} RS;

RS create_rs(float *, size_t, size_t);

void rs_step(RS);

#endif
