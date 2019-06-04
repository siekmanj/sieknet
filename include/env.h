#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdlib.h>

/*
 * Use this struct to interface an environment.
 *
 */
typedef struct env_ {
  float *state;
  void *data;
  
  size_t action_space;
  size_t observation_space;

  size_t frameskip;
  float alive_bonus;

  int *done;

  void (*dispose)(struct env_ env);
  void (*reset)(struct env_ env);
  void (*seed)(struct env_ env);
  void (*render)(struct env_ env);
  void (*close)(struct env_ env);
  float (*step)(struct env_ env, float *action);
  
} Environment;

typedef struct normalize_ {
  float *mean;
  float *mean_diff;
  float *var;
  size_t dimension;
  size_t num_steps;
  int update;
} Normalizer;

Normalizer *create_normalizer(size_t dim);

void normalize(Normalizer*, Environment*);
void save_normalizer(Normalizer*, const char *);
void dealloc_normalizer(Normalizer*);
Normalizer *load_normalizer(const char *);
Normalizer *copy_normalizer(Normalizer*);

#endif
