#include <string.h>
#include <conf.h>
#include <env.h>
#include <es.h>
#include <math.h>

#ifdef SIEKNET_USE_OMP
#include <omp.h>
#endif

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(1e-12, 1);
	float u2 = uniform(1e-12, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
	return mean + norm * std;
}

ES create_es(float (*R)(const float *, size_t, Normalizer*), float *seed, size_t num_params, size_t n){
	ES e;
	if(!seed){
		printf("Error: parameter vector cannot be null!\n");
		exit(1);
	}
	e.std = 0.0075;
	e.step_size = 0.05;
	e.num_params = num_params;
  e.num_threads = 1;

  e.f = R;

  e.params = seed;
	e.update = calloc(num_params, sizeof(float));
	e.deltas = calloc(n, sizeof(Delta*));
#ifndef SIEKNET_USE_GPU
	r.optim = cpu_init_SGD(r.params, r.update, r.num_params);
#else
  printf("ERROR: create_rs(): random search currently not supported on GPU.\n");
  exit(1);
#endif
  r.normalizer = NULL;

	for(int i = 0; i < n; i++){
		r.deltas[i] = ALLOC(Delta, 1);
		r.deltas[i]->p = ALLOC(float, num_params);
		
		for(int j = 0; j < num_params; j++)
			r.deltas[i]->p[j] = normal(0, r.std);
	}
	return r;
}
