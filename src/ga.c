/* 
 * WARNING: This file is sort of a mess and should be mostly ignored. It would
 * be very time consuming to refactor, and I would like to continue to have these
 * algorithms available, so I will be leaving it up in its current state until
 * I make some change that breaks it. You should consider everything in this file
 * strongly deprecated. Use at your own risk.
 */
#include <ga.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define PI 3.14159
#define ALPHA 1.0

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(0, 1);
	float u2 = uniform(0, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * PI * u2);
	return mean + norm * std;
}

#define SHARP_SAFETY
static float safety(float g){
  #ifdef SHARP_SAFETY
  if(ALPHA * g < 1.0)
    return 1.0;
  else
    return 1 / (ALPHA * g);
  #else
  if(ALPHA * g < 0)
    return exp(ALPHA * g);
  else
    return exp(-ALPHA * g);
  #endif
}

void baseline_recombine(float *dest, const float noise_std, const float *a, const float *b, const float mutation_rate, const size_t size){
	for(int i = 0; i < size; i++){
		if(rand()&1)
			dest[i] = a[i];
		else
			dest[i] = b[i];
		
		if(uniform(0, 1) < mutation_rate){
			dest[i] += normal(0, noise_std);
    }
	}
}

void safe_recombine(float *dest, const float noise_std, const float *a, const float *ag, const float *b, const float *bg, const float mutation_rate, const size_t size){
	for(int i = 0; i < size; i++){
		if(rand()&1){
			dest[i] = a[i];
			if(uniform(0, 1) < mutation_rate){
				dest[i] += safety(ag[i]) * normal(0, noise_std);
      }
		}else{
			dest[i] = b[i];
			if(uniform(0, 1) < mutation_rate){
				dest[i] += safety(bg[i]) * normal(0, noise_std);
      }
		}
	}
}

void aggressive_recombine(float *dest, const float noise_std, const float *a, const float *ag, const float *b, const float *bg, const float mutation_rate, const size_t size){
  for(int i = 0; i < size; i++){
    if(rand()&1){
      dest[i] = a[i];
      if(uniform(0, 1) < mutation_rate){
        dest[i] += (-safety(ag[i]) + 1) * normal(0, noise_std);
      }
    }else{
      dest[i] = b[i];
      if(uniform(0, 1) < mutation_rate){
        dest[i] += (-safety(bg[i]) + 1) * normal(0, noise_std);
      }
    }
  }
}

#define BETA 0.9f
void momentum_recombine(float *dest, const float noise_std, const float *a, const float *b, const float *momentum1, const float *momentum2, float *momentum, const float mutation_rate, const size_t size){
  for(int i = 0; i < size; i++){
    if(rand()&1){
      dest[i] = a[i];
			if(uniform(0, 1) < mutation_rate){
        float noise = normal(0, noise_std);
        dest[i] += noise + momentum1[i];
        momentum[i] = BETA * momentum1[i] + noise;
      }
    }else{
      dest[i] = b[i];
			if(uniform(0, 1) < mutation_rate){
        float noise = normal(0, noise_std);
        dest[i] += noise + momentum2[i];
        momentum[i] = BETA * momentum2[i] + noise;
      }
    }
  }
}

void safe_momentum_recombine(float *dest, const float noise_std, const float *a, const float *ag, const float *b, const float *bg, float *momentum1, float *momentum2, float *momentum, const float mutation_rate, const size_t size){
  for(int i = 0; i < size; i++){
    if(rand()&1){
      dest[i] = a[i];
			if(uniform(0, 1) < mutation_rate){
        float noise = safety(ag[i]) * normal(0, noise_std);
        dest[i] += noise + momentum1[i];
        momentum[i] = BETA * momentum1[i] + noise;
      }
    }else{
      dest[i] = b[i];
			if(uniform(0, 1) < mutation_rate){
        float noise = safety(bg[i]) * normal(0, noise_std);
        dest[i] += noise + momentum2[i];
        momentum[i] = BETA * momentum1[i] + noise;
      }
    }
  }
}

void sensitivity_gradient(float *gradient, const float *output, Nonlinearity nonl, size_t dim){
  for(int i = 0; i < dim; i++){
    gradient[i] = differentiate(output[i], nonl);
  }
}



GA create_ga(float *seed, size_t num_params, size_t size){
	GA p;

	p.mutation_type = BASELINE;
	p.size = size;

	p.mutation_rate = 0.01;
	p.noise_std = 0.01;
	p.elite_percentile = 0.95;
  p.crossover = 1;
	p.members = ALLOC(Member*, size);
  for(int i = 0; i < size; i++){
    p.members[i] = ALLOC(Member, 1);
    p.members[i]->num_params = num_params;
    p.members[i]->params     = ALLOC(float, num_params);
    p.members[i]->param_grad = calloc(num_params, sizeof(float));
    p.members[i]->momentum   = calloc(num_params, sizeof(float));

    for(int j = 0; j < num_params; j++){
      if(seed)
        p.members[i]->params[j] = seed[j];
      else
        p.members[i]->params[j] = normal(0, 0.5);
    }
  }
	return p;
}

int member_comparator(const void *a, const void *b){
	float l = (*(Member**)a)->performance;
	float r = (*(Member**)b)->performance;
  if(l < r) return 1;
  if(l > r) return -1;
  return 0;
}

void ga_sort(GA *p){
	qsort(p->members, p->size, sizeof(Member*), member_comparator);
	return;
}

void ga_breed(GA *p){
	size_t size = p->size;
  for(int i = size - (int)((p->elite_percentile)*size); i < size; i++){
    int parent1_idx = rand() % (size - (int)((p->elite_percentile)*size));
    int parent2_idx = rand() % (size - (int)((p->elite_percentile)*size));
    Member *a = p->members[parent1_idx];
    Member *b = p->crossover ? p->members[parent2_idx] : a;

    Member *child = p->members[i];
    switch(p->mutation_type){
      case NONE:
        baseline_recombine(child->params, p->noise_std, a->params, b->params, 0.0, a->num_params);
        break;
      case BASELINE:
        baseline_recombine(child->params, p->noise_std, a->params, b->params, p->mutation_rate, a->num_params);
        break;
      case MOMENTUM:
        momentum_recombine(child->params, p->noise_std, a->params, b->params, a->momentum, b->momentum, child->momentum, p->mutation_rate, a->num_params);
        break;
      case SAFE:
        safe_recombine(child->params, p->noise_std, a->params, a->param_grad, b->params, b->param_grad, p->mutation_rate, a->num_params);
        break;
      case AGGRESSIVE:
        aggressive_recombine(child->params, p->noise_std, a->params, a->param_grad, b->params, b->param_grad, p->mutation_rate, a->num_params);
        break;
      case SAFE_MOMENTUM:
        safe_momentum_recombine(child->params, p->noise_std, a->params, a->param_grad, b->params, b->param_grad, a->momentum, b->momentum, child->momentum, p->mutation_rate, a->num_params);
        break;
    }

	}
  for(int i = 0; i < size; i++){
    memset(p->members[i]->param_grad, '\0', p->members[i]->num_params);
  }
}

void ga_evolve(GA *p){
  ga_sort(p);
  ga_breed(p);
}

