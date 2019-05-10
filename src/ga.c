#include <ga.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define PI 3.14159
#define ALPHA 1.0

#ifdef SIEKNET_USE_GPU
#error "ERROR: Use of genetic algorithms is currently not supported on the GPU."
#endif

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(0, 1);
	float u2 = uniform(0, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * PI * u2);
	return mean + norm * std;
}

static float safety(float g){
  if(ALPHA * g < 1.0)
    return 1.0;
  else
    return 1 / (ALPHA * g);
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

#define BETA 0.99f
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

void recombine(float *dest, const float noise_std, const Mutation_type type, const float *params1, const float *paramgrad1, const float *params2, const float *paramgrad2, float *momentum1, float *momentum2, float *momentum, const float mutation_rate, const size_t num_params){
  switch(type){
    case NONE:
      return baseline_recombine(dest, noise_std, params1, params2, 0.0, num_params);
      break;
    case BASELINE:
      return baseline_recombine(dest, noise_std, params1, params2, mutation_rate, num_params);
      break;
    case MOMENTUM:
      return momentum_recombine(dest, noise_std, params1, params2, momentum1, momentum2, momentum, mutation_rate, num_params);
      break;
    case SAFE:
      return safe_recombine(dest, noise_std, params1, paramgrad1, params2, paramgrad2, mutation_rate, num_params);
      break;
    case SAFE_MOMENTUM:
      return safe_momentum_recombine(dest, noise_std, params1, paramgrad1, params2, paramgrad2, momentum1, momentum2, momentum, mutation_rate, num_params);
      break;
  }
}

void sensitivity_gradient(float *gradient, const float *output, Nonlinearity nonl, size_t dim){
  for(int i = 0; i < dim; i++){
    gradient[i] = differentiate(output[i], nonl);
  }
}



Pool create_pool(float *seed, size_t num_params, size_t pool_size){
	Pool p;

	p.mutation_type = BASELINE;
	p.pool_size = pool_size;

	p.mutation_rate = 0.01;
	p.noise_std = 0.01;
	p.elite_percentile = 0.95;
  p.crossover = 1;
	p.members = ALLOC(Member*, pool_size);
  for(int i = 0; i < pool_size; i++){
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
      //printf("p[%d][%d]: %f???\n", i, j, p.members[i]->params[j]);
    }
    //for(int j = 0; j < num_params && i && seed; j++)
      //p.members[i]->params[j] += normal(0, 0.5);
  }
  //printf("FUCK random:\n");
  //PRINTLIST(p.members[3]->params, p.members[3]->num_params);
	return p;
}

int member_comparator(const void *a, const void *b){
	float l = (*(Member**)a)->performance;
	float r = (*(Member**)b)->performance;
  if(l < r) return 1;
  if(l > r) return -1;
  return 0;
}

void sort_pool(Pool *p){
	qsort(p->members, p->pool_size, sizeof(Member*), member_comparator);
	return;
}

void breed_pool(Pool *p){
	size_t size = p->pool_size;
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
      case SAFE_MOMENTUM:
        safe_momentum_recombine(child->params, p->noise_std, a->params, a->param_grad, b->params, b->param_grad, a->momentum, b->momentum, child->momentum, p->mutation_rate, a->num_params);
        break;
    }

	}
  for(int i = 0; i < size; i++){
    memset(p->members[i]->param_grad, '\0', p->members[i]->num_params);
  }
}

void evolve_pool(Pool *p){
  sort_pool(p);
  breed_pool(p);
}

