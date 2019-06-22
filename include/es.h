#ifndef EVOLUTIONARY_STRATEGIES_H
#define EVOLUTIONARY_STRATEGIES_H

#include <stdlib.h>
#include <stdio.h>
#include <optimizer.h>

typedef struct eps_{
  float *p;
  float r;
} Epsilon;

typedef struct es_{
  float std;
  float step_size;
  
  size_t n;
  size_t num_params;
  size_t num_threads;

  float *params;
  float *update;
  Epsilon **eps;
  float (*f)(const float *, size_t, Normalizer*);

  Normalizer *normalizer;
  SGD optim;
} ES;

ES create_es(float (*R)(const float *, size_t, Normalizer*), float *, size_t, size_t);

void es_step(ES);

#endif
