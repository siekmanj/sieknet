#ifndef RANDOM_SEARCH_H
#define RANDOM_SEARCH_H

#include <stdlib.h>
#include <stdio.h>

typedef enum search_type_t {v1, v1_t, v2, v2_t} Search_type;

/*
typedef struct perturbance_{
	float *p;
	float r_pos;
	float r_neg;
} Perturbance;
*/

typedef struct RS{
	float std;
	float step_size;
	size_t directions;
	size_t num_params;

	float *params;
	float **perturbances;

	float *r_pos;
	float *r_neg;

	Search_type algo;
} RS;

RS create_rs(float *, size_t, size_t);

void rs_step(RS);

#endif
