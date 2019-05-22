#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <env.h>
#include <conf.h>

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(0, 1);
	float u2 = uniform(0, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * 3.14159 * u2);
	return mean + norm * std;
}

Normalizer *create_normalizer(size_t dim){
  Normalizer *n = (Normalizer*)malloc(sizeof(Normalizer));
  n->dimension = dim;
  n->env_mean = calloc(n->dimension, sizeof(float));
  n->env_std = calloc(n->dimension, sizeof(float));

  for(int i = 0; i < n->dimension; i++){
    n->env_mean[i] = 0.0f;
    n->env_std[i] = 1.0f;
  }

  n->num_steps = 0;
  return n;
}

void dealloc_normalizer(Normalizer n){
  free(n.env_mean);
  free(n.env_std);
}

Normalizer copy_normalizer(Normalizer n){
  Normalizer ret;
  ret.dimension = n.dimension;
  ret.num_steps = n.num_steps;
  ret.update = n.update;
  ret.env_mean = malloc(sizeof(float) * n.dimension);
  ret.env_std  = malloc(sizeof(float) * n.dimension);
  for(int i = 0; i < n.dimension; i++){
    ret.env_mean[i] = n.env_mean[i];
    ret.env_std[i] = n.env_std[i];
  }
  return ret;
}

void save_normalizer(Normalizer n, const char *filename){
  FILE *fp = fopen(filename, "wb");
  if(!fp){
    printf("ERROR: save_normalizer(): unable to open '%s' for write.\n", filename);
    exit(1);
  }
  fprintf(fp, "%lu %lu ", n.dimension, n.num_steps);
  for(int i = 0; i < n.dimension; i++)
    fprintf(fp, "%f %f", n.env_mean[i], n.env_std[i]);
}

Normalizer load_normalizer(const char *filename){
  FILE *fp = fopen(filename, "rb");
  if(!fp){
    printf("ERROR: load_normalizer(): unable to open '%s'.\n", filename);
    exit(1);
  }
  Normalizer n;
  if(fscanf(fp, "%lu %lu ", &n.dimension, &n.num_steps) == EOF){
    printf("ERROR: load_normalizer(): EOF reached while loading file - probably corrupted.\n");
    exit(1);
  }
  n.env_mean = malloc(sizeof(float) * n.dimension);
  n.env_std  = malloc(sizeof(float) * n.dimension);
  n.env_var  = malloc(sizeof(float) * n.dimension);
  for(int i = 0; i < n.dimension; i++)
    if(fscanf(fp, "%f %f", &n.env_mean[i], &n.env_std[i]) == EOF){
      printf("ERROR: load_normalizer(): EOF reached while loading file - probably corrupted.\n");
      exit(1);
    }

  return n;
}

void normalize(Normalizer *n, Environment *e){
  if(!n)
    return;

  /* Using Welford's algorithm */
  if(n->update){
    n->num_steps++;
    for(int i = 0; i < n->dimension; i++){
      /* Calculate running mean */
      float delta1 = e->state[i] - n->env_mean[i];
      n->env_mean[i] += delta1 / n->num_steps;
      float delta2 = e->state[i] - n->env_mean[i];


      /* Calculate running variance */
      n->env_std[i] += sqrt(delta1 * delta2 / n->num_steps);
    }
  }

  /* Normalize environment state */
  for(int i = 0; i < n->dimension; i++){
    e->state[i] = (e->state[i] - n->env_mean[i]) / n->env_std[i];
  }
}
