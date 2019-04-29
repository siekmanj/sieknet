#include <ga.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define PI 3.14159

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
	float alpha = 1.0;
	if(g < 0)
		return exp(alpha*g);
	else
		return exp(-alpha*g);
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
	}
  return ret;
}

float *momentum_recombine(const float step_size, const float *a, const float *b, const float *momentum, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *safe_momentum_recombine(const float step_size, const float *a, const float *ag, const float *b, const float *bg, float *momentum, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *recombine(const float step_size, const Mutation_type type, const float *params1, const float *paramgrad1, const float *params2, const float *paramgrad2, float *momentum, const float mutation_rate, const size_t num_params){
  switch(type){
    case NONE:
      return baseline_recombine(step_size, params1, params2, 0.0, num_params);
      break;
    case BASELINE:
      return baseline_recombine(step_size, params1, params2, mutation_rate, num_params);
      break;
    case MOMENTUM:
      return momentum_recombine(step_size, params1, params2, momentum, mutation_rate, num_params);
      break;
    case SAFE:
      return safe_recombine(step_size, params1, paramgrad1, params2, paramgrad2, mutation_rate, num_params);
      break;
    case SAFE_MOMENTUM:
      return safe_momentum_recombine(step_size, params1, paramgrad1, params2, paramgrad2, momentum, mutation_rate, num_params);
      break;
  }
  return NULL;
}

void sensitivity_gradient(float *gradient, const float *output, Nonlinearity nonl, size_t dim){
  for(int i = 0; i < dim; i++){
    gradient[i] = differentiate(output[i], nonl);
  }
}

#define fill_comparator(type, a, b)       \
  float l = (*((type **)a))->performance; \
  float r = (*((type **)b))->performance; \
  if(l < r) return 1;                     \
  if(l > r) return -1;                    \
  return 0

int mlp_comparator(const void *a, const void *b){
  fill_comparator(MLP, a, b);
}

int rnn_comparator(const void *a, const void *b){
  fill_comparator(RNN, a, b);
}

int lstm_comparator(const void *a, const void *b){
  fill_comparator(LSTM, a, b);
}


Pool create_pool(Network_type type, void *seed, size_t pool_size){
	Pool p;
	p.network_type = type;
	p.mutation_type = BASELINE;
	p.pool_size = pool_size;

	p.mutation_rate = 0.01;
	p.step_size = 0.01;
	p.elite_percentile = 0.95;

	switch(type){
		case mlp:
			p.members = (void*)ALLOC(MLP *, pool_size);
			p.members[0] = copy_mlp((MLP*)seed);
			p.momentum = ALLOC(float, ((MLP*)seed)->num_params);

			for(int i = 1; i < pool_size; i++){
				p.members[i] = copy_mlp((MLP*)seed);
				for(int j = 0; j < ((MLP*)seed)->num_params; j++)
					((MLP*)p.members[i])->params[j] = normal(0, 0.5);
			}
			break;

		case rnn:
			p.members = (void*)ALLOC(RNN *, pool_size);
			p.members[0] = copy_rnn((RNN*)seed);
			p.momentum = ALLOC(float, ((RNN*)seed)->num_params);

			for(int i = 1; i < pool_size; i++){
				p.members[i] = copy_rnn((RNN*)seed);
				for(int j = 0; j < ((RNN*)seed)->num_params; j++)
					((RNN*)p.members[i])->params[j] = normal(0, 0.5);
			}
			break;

		case lstm:
			p.members = (void*)ALLOC(LSTM *, pool_size);
			p.members[0] = copy_lstm((LSTM*)seed);
			p.momentum = ALLOC(float, ((LSTM*)seed)->num_params);

			for(int i = 1; i < pool_size; i++){
				p.members[i] = copy_lstm((LSTM*)seed);
				for(int j = 0; j < ((LSTM*)seed)->num_params; j++)
					((LSTM*)p.members[i])->params[j] = normal(0, 0.5);
			}
			break;
	}
	return p;
}

void sort_pool(Pool *p){
	switch(p->network_type){
		case mlp:
			qsort(p->members, p->pool_size, sizeof(MLP*), mlp_comparator);
			break;
		
		case rnn:
			qsort(p->members, p->pool_size, sizeof(RNN*), rnn_comparator);
			break;

		case lstm:
			qsort(p->members, p->pool_size, sizeof(LSTM*), lstm_comparator);
			break;
	}
	return;
}

void cull_pool(Pool *p){
	size_t size = p->pool_size;
	float percentile = p->elite_percentile;

	switch(p->network_type){
		case mlp:
			for(int i = size - (int)((percentile)*size); i < size; i++){
				dealloc_mlp((MLP*)p->members[i]);
				p->members[i] = NULL;
			}
			break;
		
		case rnn:
			for(int i = size - (int)((percentile)*size); i < size; i++){
				dealloc_rnn((RNN*)p->members[i]);
				p->members[i] = NULL;
			}
			
			break;

		case lstm:
			for(int i = size - (int)((percentile)*size); i < size; i++){
				dealloc_lstm((LSTM*)p->members[i]);
				p->members[i] = NULL;
			}
			break;
	}
}

void breed_pool(Pool *p){
	size_t size = p->pool_size;
	switch(p->network_type){
		case mlp:
			for(int i = size - (int)((p->elite_percentile)*size); i < size; i++){
				MLP *a = (MLP*)p->members[rand() % (size - (int)((p->elite_percentile)*size))];
				MLP *b = (MLP*)p->members[rand() % (size - (int)((p->elite_percentile)*size))];

				MLP *child = copy_mlp(a);
				child->params = recombine(p->step_size, p->mutation_type, a->params, a->param_grad, b->params, b->param_grad, p->momentum, p->mutation_rate, a->num_params);
				p->members[i] = (void*)child;
			}
			break;
		case rnn:
			for(int i = size - (int)((p->elite_percentile)*size); i < size; i++){
				RNN *a = (RNN*)p->members[rand() % (size - (int)((p->elite_percentile)*size))];
				RNN *b = (RNN*)p->members[rand() % (size - (int)((p->elite_percentile)*size))];

				RNN *child = copy_rnn(a);
				child->params = recombine(p->step_size, p->mutation_type, a->params, a->param_grad, b->params, b->param_grad, p->momentum, p->mutation_rate, a->num_params);
				p->members[i] = (void*)child;
			}
			break;
		case lstm:
			for(int i = size - (int)((p->elite_percentile)*size); i < size; i++){
				LSTM *a = (LSTM*)p->members[rand() % (size - (int)((p->elite_percentile)*size))];
				LSTM *b = (LSTM*)p->members[rand() % (size - (int)((p->elite_percentile)*size))];

				LSTM *child = copy_lstm(a);
				child->params = recombine(p->step_size, p->mutation_type, a->params, a->param_grad, b->params, b->param_grad, p->momentum, p->mutation_rate, a->num_params);
				p->members[i] = (void*)child;
			}
			break;
	}
}

void evolve_pool(Pool *p){
  sort_pool(p);
  cull_pool(p);
  breed_pool(p);
}

