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
  n->env_std  = calloc(n->dimension, sizeof(float));
  n->env_var  = calloc(n->dimension, sizeof(float));

  for(int i = 0; i < n->dimension; i++){
    n->env_mean[i] = 0.0f;
    n->env_std[i] = 1.0f;
    n->env_var[i] = 1.0f;
  }

  n->num_steps = 0;
  return n;
}

void dealloc_normalizer(Normalizer *n){
  free(n->env_mean);
  free(n->env_std);
  free(n->env_var);
  free(n);
}

Normalizer *copy_normalizer(Normalizer *n){
  Normalizer *ret = create_normalizer(n->dimension);
  ret->dimension = n->dimension;
  ret->num_steps = n->num_steps;
  ret->update = n->update;
  for(int i = 0; i < n->dimension; i++){
    ret->env_mean[i] = n->env_mean[i];
    ret->env_std[i] = n->env_std[i];
    ret->env_var[i] = n->env_var[i];
  }
  return ret;
}

void save_normalizer(Normalizer *n, const char *filename){
  FILE *fp = fopen(filename, "wb");
  if(!fp){
    printf("ERROR: save_normalizer(): unable to open '%s' for write.\n", filename);
    exit(1);
  }
  fprintf(fp, "%lu %lu ", n->dimension, n->num_steps);
  for(int i = 0; i < n->dimension; i++)
    fprintf(fp, "%f %f", n->env_mean[i], n->env_std[i]);
}

Normalizer *load_normalizer(const char *filename){
  FILE *fp = fopen(filename, "rb");
  if(!fp)
    return NULL;
  
  size_t dim, num_steps;
  if(fscanf(fp, "%lu %lu ", &dim, &num_steps) == EOF){
    printf("ERROR: load_normalizer(): EOF reached while loading file - probably corrupted.\n");
    exit(1);
  }
  Normalizer *n = create_normalizer(dim);
  n->num_steps = num_steps;
  for(int i = 0; i < n->dimension; i++){
    if(fscanf(fp, "%f %f", &n->env_mean[i], &n->env_std[i]) == EOF){
      printf("ERROR: load_normalizer(): EOF reached while loading file - probably corrupted.\n");
      exit(1);
    }
  }

  return n;
}

void normalize(Normalizer *n, Environment *e){
  if(!n)
    return;

  /* Using Welford's algorithm */
  /* https://www.johndcook.com/blog/standard_deviation */
  if(n->update){
    n->num_steps++;
    for(int i = 0; i < n->dimension; i++){
      float old_mean = n->env_mean[i];
      float old_var  = n->env_var[i];
      float new_mean, new_var;
      
      if(n->num_steps == 1){
        n->env_mean[i] = e->state[i];
        n->env_var[i] = 0.0f;
        n->env_std[i] = 1.0f;
      }else{
        new_mean = old_mean + (e->state[i] - old_mean) / n->num_steps;
        new_var  = old_var  + (e->state[i] - old_mean) * (e->state[i] - new_mean);

        n->env_mean[i] = new_mean;
        n->env_var[i]  = new_var;
        n->env_std[i]  = sqrt(new_var);
      }
    }
  }

  /* Normalize environment state */
  for(int i = 0; i < n->dimension; i++){
    e->state[i] = (e->state[i] - n->env_mean[i]) / n->env_std[i];
  }
}
