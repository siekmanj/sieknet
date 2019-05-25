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
  n->mean = calloc(n->dimension, sizeof(float));
  n->var  = calloc(n->dimension, sizeof(float));
  n->mean_diff  = calloc(n->dimension, sizeof(float));

  for(int i = 0; i < n->dimension; i++){
    n->mean[i] = 0.0f;
    n->var[i] = 1.0f;
    n->mean_diff[i] = 1.0f;
  }

  n->num_steps = 0;
  return n;
}

void dealloc_normalizer(Normalizer *n){
  free(n->mean);
  free(n->mean_diff);
  free(n->var);
  free(n);
}

Normalizer *copy_normalizer(Normalizer *n){
  Normalizer *ret = create_normalizer(n->dimension);
  ret->dimension = n->dimension;
  ret->num_steps = n->num_steps;
  ret->update = n->update;
  for(int i = 0; i < n->dimension; i++){
    ret->mean[i] = n->mean[i];
    ret->var[i] = n->var[i];
    ret->mean_diff[i] = n->mean_diff[i];
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
    fprintf(fp, "%f %f ", n->mean[i], n->mean_diff[i]);
  fclose(fp);
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
    if(fscanf(fp, "%f %f", &n->mean[i], &n->mean_diff[i]) == EOF){
      printf("ERROR: load_normalizer(): EOF reached while loading file - probably corrupted.\n");
      exit(1);
    }
  }
  fclose(fp);

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
      if(n->num_steps == 1){
        n->mean[i] = e->state[i];
        n->mean_diff[i] = 0.0f;
        n->var[i] = 1.0f;
      }else{
				float old_mean = n->mean[i];
				n->mean[i]      += (e->state[i] - n->mean[i]) / n->num_steps;
				n->mean_diff[i] += (e->state[i] - old_mean) * (e->state[i] - n->mean[i]);
				n->var[i] = n->mean_diff[i] / n->num_steps;
      }
    }
  }

  /* Normalize environment state */
  for(int i = 0; i < n->dimension; i++){
    e->state[i] = (e->state[i] - n->mean[i]) / sqrt(n->var[i]);
  }
}
