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

MLP *copy_mlp(MLP *n){
	size_t arr[n->depth+1];
	arr[0] = n->input_dimension;
	for(int i = 0; i < n->depth; i++){
		arr[i+1] = n->layers[i].size;
	}
	MLP *ret = ALLOC(MLP, 1);
	*ret = mlp_from_arr(arr, n->depth+1);

	for(int i = 0; i < n->depth; i++){
		ret->layers[i].logistic = n->layers[i].logistic;
	}

	for(int i = 0; i < n->num_params; i++)
		ret->params[i] = n->params[i];
  ret->performance = 0;

  return ret;
}

RNN *copy_rnn(RNN *n){
	size_t arr[n->depth+2];
	arr[0] = n->input_dimension;
	for(int i = 0; i < n->depth; i++){
		arr[i+1] = n->layers[i].size;
	}
	arr[n->depth+1] = n->output_layer.size;

	RNN *ret = ALLOC(RNN, 1);
	*ret = rnn_from_arr(arr, n->depth+2);

	for(int i = 0; i < n->depth; i++){
		ret->layers[i].logistic = n->layers[i].logistic;
	}
	ret->output_layer.logistic = n->output_layer.logistic;

	for(int i = 0; i < n->num_params; i++)
		ret->params[i] = n->params[i];
  ret->performance = 0;

  return ret;
}

LSTM *copy_lstm(LSTM *n){
	size_t arr[n->depth+2];
	arr[0] = n->input_dimension;
	for(int i = 0; i < n->depth; i++){
		arr[i+1] = n->layers[i].size;
	}
	arr[n->depth+1] = n->output_layer.size;

	LSTM *ret = ALLOC(LSTM, 1);
	*ret = lstm_from_arr(arr, n->depth+2);

	ret->output_layer.logistic = n->output_layer.logistic;

	for(int i = 0; i < n->num_params; i++)
		ret->params[i] = n->params[i];
  ret->performance = 0;

  return ret;
}

static float safety(float g){
	return 1 / (ALPHA * g);
}


float *baseline_recombine(const float step_size, const float *a, const float *b, const float mutation_rate, const size_t size){
  float *ret = ALLOC(float, size);
	for(int i = 0; i < size; i++){
		if(rand()&1)
			ret[i] = a[i];
		else
			ret[i] = b[i];
		
		if(uniform(0, 1) < mutation_rate)
			ret[i] += normal(0, step_size);
	}
  return ret;
}

float *safe_recombine(const float step_size, const float *a, const float *ag, const float *b, const float *bg, const float mutation_rate, const size_t size){
	float *ret = ALLOC(float, size);
	for(int i = 0; i < size; i++){
		if(rand()&1){
			ret[i] = a[i];
			if(uniform(0, 1) < mutation_rate)
				ret[i] += safety(ag[i]) * normal(0, step_size);
		}else{
			ret[i] = b[i];
			if(uniform(0, 1) < mutation_rate)
				ret[i] += safety(bg[i]) * normal(0, step_size);
		}
		if(isnan(ret[i]))
			printf("got nan!\n");
	}
  return ret;
}

float *momentum_recombine(const float step_size, const float *a, const float *b, float *momentum1, float *momentum2, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *safe_momentum_recombine(const float step_size, const float *a, const float *ag, const float *b, const float *bg, float *momentum1, float *momentum2, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *recombine(const float step_size, const Mutation_type type, const float *params1, const float *paramgrad1, const float *params2, const float *paramgrad2, float *momentum1, float *momentum2, const float mutation_rate, const size_t num_params){
  switch(type){
    case NONE:
      return baseline_recombine(step_size, params1, params2, 0.0, num_params);
      break;
    case BASELINE:
      return baseline_recombine(step_size, params1, params2, mutation_rate, num_params);
      break;
    case MOMENTUM:
      return momentum_recombine(step_size, params1, params2, momentum1, momentum2, mutation_rate, num_params);
      break;
    case SAFE:
      return safe_recombine(step_size, params1, paramgrad1, params2, paramgrad2, mutation_rate, num_params);
      break;
    case SAFE_MOMENTUM:
      return safe_momentum_recombine(step_size, params1, paramgrad1, params2, paramgrad2, momentum1, momentum2, mutation_rate, num_params);
      break;
  }
  return NULL;
}

void sensitivity_gradient(float *gradient, const float *output, Nonlinearity nonl, size_t dim){
  for(int i = 0; i < dim; i++){
    gradient[i] = differentiate(output[i], nonl);
  }
}



Pool create_pool(Network_type type, void *seed, size_t pool_size){
	Pool p;
	p.network_type = type;
	p.mutation_type = BASELINE;
	p.pool_size = pool_size;

	p.mutation_rate = 0.01;
	p.step_size = 0.01;
	p.elite_percentile = 0.95;
  p.crossover = 1;
	p.members = ALLOC(Member*, pool_size);
	switch(type){
		case mlp:
			p.members[0] = ALLOC(Member, 1);
			p.members[0]->network = copy_mlp((MLP*)seed);
			p.momentum = ALLOC(float, ((MLP*)seed)->num_params);

			for(int i = 1; i < pool_size; i++){
				p.members[i] = ALLOC(Member, 1);
				p.members[i]->network = copy_mlp((MLP*)seed);
				for(int j = 0; j < ((MLP*)seed)->num_params; j++)
					((MLP*)p.members[i]->network)->params[j] = normal(0, 0.5);
			}
			break;

		case rnn:
			p.members[0]->network = copy_rnn((RNN*)seed);
			p.momentum = ALLOC(float, ((RNN*)seed)->num_params);

			for(int i = 1; i < pool_size; i++){
				p.members[i]->network = copy_rnn((RNN*)seed);
				for(int j = 0; j < ((RNN*)seed)->num_params; j++)
					((RNN*)p.members[i]->network)->params[j] = normal(0, 0.5);
			}
			break;

		case lstm:
			p.members[0]->network = copy_lstm((LSTM*)seed);
			p.momentum = ALLOC(float, ((LSTM*)seed)->num_params);

			for(int i = 1; i < pool_size; i++){
				p.members[i]->network = copy_lstm((LSTM*)seed);
				for(int j = 0; j < ((LSTM*)seed)->num_params; j++)
					((LSTM*)p.members[i]->network)->params[j] = normal(0, 0.5);
			}
			break;
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

void sort_pool(Pool *p){
	qsort(p->members, p->pool_size, sizeof(Member*), member_comparator);
	return;
}

void cull_pool(Pool *p){
	size_t size = p->pool_size;
	float percentile = p->elite_percentile;

	for(int i = size - (int)((percentile)*size); i < size; i++){
		void *n = p->members[i]->network;
		switch(p->network_type){
			case mlp:
				dealloc_mlp((MLP*)n);
				break;
			
			case rnn:
				dealloc_rnn((RNN*)n);
				break;

			case lstm:
				dealloc_lstm((LSTM*)n);
				break;
		}
		p->members[i]->performance = 0;
		//free(p->members[i]->momentum);
	}
}

void breed_pool(Pool *p){
	size_t size = p->pool_size;
  for(int i = size - (int)((p->elite_percentile)*size); i < size; i++){
    int parent1_idx = rand() % (size - (int)((p->elite_percentile)*size));
    int parent2_idx = rand() % (size - (int)((p->elite_percentile)*size));
    Member *parent1 = p->members[parent1_idx];
    Member *parent2 = p->crossover ? p->members[parent2_idx] : parent1;

    //printf("parent 1: %p, parent 2: %p\n", parent1, parent2);

    //printf("\nMaking new child from parents %d and %d! rand() %% (%d - %d)\n", parent1_idx, parent2_idx, (int)size, (int)((p->elite_percentile)*size));
    switch(p->network_type){
      case mlp:
        {
          MLP *a = (MLP*)parent1->network;
          MLP *b = (MLP*)parent2->network;

          MLP *child = copy_mlp(a);
          child->params = recombine(p->step_size, p->mutation_type, a->params, a->param_grad, b->params, b->param_grad, parent1->momentum, parent2->momentum, p->mutation_rate, a->num_params);
          p->members[i]->network = (void*)child;
        }
        break;
      case rnn:
        {
          RNN *a = (RNN*)p->members[parent1_idx]->network;
          RNN *b = (RNN*)p->members[parent2_idx]->network;

          RNN *child = copy_rnn(a);
          child->params = recombine(p->step_size, p->mutation_type, a->params, a->param_grad, b->params, b->param_grad, parent1->momentum, parent2->momentum, p->mutation_rate, a->num_params);
          p->members[i]->network = (void*)child;
        }
        break;
      case lstm:
        {
          LSTM *a = (LSTM*)p->members[parent1_idx]->network;
          LSTM *b = (LSTM*)p->members[parent2_idx]->network;

          LSTM *child = copy_lstm(a);
          child->params = recombine(p->step_size, p->mutation_type, a->params, a->param_grad, b->params, b->param_grad, parent1->momentum, parent2->momentum, p->mutation_rate, a->num_params);
          p->members[i]->network = (void*)child;
        }
        break;
    }
  }
	// If we use SAFE or SAFE_MOMENTUM, clear the parameter gradients of the elites 
	if(p->mutation_type == SAFE || p->mutation_type == SAFE_MOMENTUM){
		for(int i = 0; i < size - (int)((p->elite_percentile)*size); i++){
			switch(p->network_type){
				case mlp:
					{
						MLP *m = (MLP*)p->members[i]->network;
						memset(m->param_grad, '\0', m->num_params*sizeof(float));
					}
					break;
				case rnn:
					{
						RNN *m = (RNN*)p->members[i]->network;
						memset(m->param_grad, '\0', m->num_params*sizeof(float));
					}
					break;
				case lstm:
					{
						LSTM *m = (LSTM*)p->members[i]->network;
						memset(m->param_grad, '\0', m->num_params*sizeof(float));
					}
					break;
			}
		}
	}
}

void evolve_pool(Pool *p){
  sort_pool(p);
  cull_pool(p);
  breed_pool(p);
}

