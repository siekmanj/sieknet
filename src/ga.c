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
    case MUT_none:
      return baseline_recombine(step_size, params1, params2, 0.0, num_params);
      break;
    case MUT_baseline:
      return baseline_recombine(step_size, params1, params2, mutation_rate, num_params);
      break;
    case MUT_momentum:
      return momentum_recombine(step_size, params1, params2, momentum, mutation_rate, num_params);
      break;
    case MUT_safe:
      return safe_recombine(step_size, params1, paramgrad1, params2, paramgrad2, mutation_rate, num_params);
      break;
    case MUT_safe_momentum:
      return safe_momentum_recombine(step_size, params1, paramgrad1, params2, paramgrad2, momentum, mutation_rate, num_params);
      break;
  }
  return NULL;
}

/* forgive me for I have sinned */

#define agnostic_recombine(TYPE, type, m, a, b) \
  assert(a->num_params == b->num_params); \
	if(!m.momentum && (m.mutation_type == MUT_momentum || m.mutation_type == MUT_safe_momentum)){ \
		m.momentum = ALLOC(float, a->num_params);                                                   \
		memset(m.momentum, '\0', sizeof(float)*a->num_params);                                      \
	}                                                                                             \
  TYPE *ret = copy_ ## type (a);                                                                \
  free(ret->params);                                                                            \
  ret->params = recombine(m.step_size, m.mutation_type, a->params, a->param_grad, b->params, b->param_grad, m.momentum, m.mutation_rate, a->num_params); \
  return ret

MLP *MLP_recombine(Mutator m, MLP *a, MLP *b){
	agnostic_recombine(MLP, mlp, m, a, b);
}

RNN *RNN_recombine(Mutator m, RNN *a, RNN *b){
	agnostic_recombine(RNN, rnn, m, a, b);
}

LSTM *LSTM_recombine(Mutator m, LSTM *a, LSTM *b){
	agnostic_recombine(LSTM, lstm, m, a, b);
}

/* don't try this at home kids  */

#define fill_pool(type, p, size, seed, random) \
	for(int i = 0; i < size; i++){               \
		p[i] = copy_ ## type (seed);               \
		if(random){                                \
			for(int j = 0; j < seed->num_params; j++)\
				p[i]->params[j] = normal(0, 0.5);      \
		}                                          \
		else if(i){                                \
			for(int j = 0; j < seed->num_params; j++)\
				p[i]->params[j] += normal(0, 0.05);    \
		}                                          \
	}                                            \

MLP_pool create_mlp_pool(size_t size, MLP *seed, int random){
	MLP_pool p = ALLOC(MLP *, size);
	fill_pool(mlp, p, size, seed, random);
	return p;
}

RNN_pool create_rnn_pool(size_t size, RNN *seed, int random){
	RNN_pool p = ALLOC(RNN *, size);
	fill_pool(rnn, p, size, seed, random);
	return p;
}

LSTM_pool create_lstm_pool(size_t size, LSTM *seed, int random){
	LSTM_pool p = ALLOC(LSTM *, size);
	fill_pool(lstm, p, size, seed, random);
	return p;
}

#define fill_comparator(type, a, b)       \
  float l = (*((type **)a))->performance; \
  float r = (*((type **)b))->performance; \
  if(l < r) return 1;                     \
  if(l > r) return -1;                    \
  if(l == r) return 0

int mlp_comparator(const void *a, const void *b){
  fill_comparator(MLP, a, b);
}

int rnn_comparator(const void *a, const void *b){
  fill_comparator(RNN, a, b);
}

int lstm_comparator(const void *a, const void *b){
  fill_comparator(LSTM, a, b);
}

void sort_mlp_pool(MLP_pool p, size_t size){
  qsort(p, size, sizeof(MLP*), mlp_comparator);
}

void sort_rnn_pool(RNN_pool p, size_t size){
  qsort(p, size, sizeof(RNN*), rnn_comparator);

}
void sort_lstm_pool(LSTM_pool p, size_t size){
  printf("pool[0]: %p, pool[1]: %p\n", p[0], p[1]);
  qsort(p, size, sizeof(LSTM*), lstm_comparator);
}

#define cull_pool(type, p, percentile, size)                \
  for(int i = (int)((1-percentile)*size); i < size; i++){   \
    dealloc_ ## type(p[i]);                                 \
    p[i] = NULL;                                            \
  }

void cull_mlp_pool(MLP_pool p, Mutator m, size_t size){
  cull_pool(mlp, p, m.elite_percentile, size);
}

void cull_rnn_pool(RNN_pool p, Mutator m, size_t size){
  cull_pool(rnn, p, m.elite_percentile, size);
}

void cull_lstm_pool(LSTM_pool p, Mutator m, size_t size){
  cull_pool(lstm, p, m.elite_percentile, size);
}

#define breed_pool(TYPE, p, mutator, size)                            \
  for(int i = (int)((1-m.elite_percentile)*size); i < size; i++){     \
    TYPE *parent1 = p[rand() % ((int)((1-m.elite_percentile)*size))]; \
    TYPE *parent2 = p[rand() % ((int)((1-m.elite_percentile)*size))]; \
    p[i] = (TYPE*) m.recombine(m, parent1, parent2);                  \
  }

void breed_mlp_pool(MLP_pool p, Mutator m, size_t size){
  breed_pool(MLP, p, m, size);
}

void breed_rnn_pool(RNN_pool p, Mutator m, size_t size){
  breed_pool(RNN, p, m, size);
}

void breed_lstm_pool(LSTM_pool p, Mutator m, size_t size){
  breed_pool(LSTM, p, m, size);
}

Mutator create_mutator(Network_type net, Mutation_type mut){
  Mutator m;
  m.network_type = net;
  m.mutation_type = mut;

  m.mutation_rate = 0.01;
	m.step_size = 0.05;
  m.elite_percentile = 0.9;
  m.momentum = NULL;

  switch(net){
    case mlp:
      m.recombine = (void*)MLP_recombine;
      break;
    case rnn:
      m.recombine = (void*)RNN_recombine;
      break;
    case lstm:
      m.recombine = (void*)LSTM_recombine;
      break;
  }
	return m;
}
