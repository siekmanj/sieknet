#include <ga.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifdef SIEKNET_USE_GPU
#error "ERROR: Use of genetic algorithms is currently not supported on the GPU."
#endif

MLP *copy_mlp(MLP *n){

  return NULL;
}

RNN *copy_rnn(RNN *n){

  return NULL;
}

LSTM *copy_lstm(LSTM *n){

  return NULL;
}


float *baseline_recombine(const float *a, const float *b, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *safe_recombine(const float *a, const float *ag, const float *b, const float *bg, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *momentum_recombine(const float *a, const float *b, const float *momentum, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *safe_momentum_recombine(const float *a, const float *ag, const float *b, const float *bg, float *momentum, const float mutation_rate, const size_t size){
  //TODO
  return NULL;
}

float *recombine(const Mutation_type type, const float *params1, const float *paramgrad1, const float *params2, const float *paramgrad2, float *momentum, const float mutation_rate, const size_t num_params){
  switch(type){
    case MUT_none:
      return baseline_recombine(params1, params2, 0.0, num_params);
      break;
    case MUT_baseline:
      return baseline_recombine(params1, params2, mutation_rate, num_params);
      break;
    case MUT_momentum:
      return momentum_recombine(params1, params2, momentum, mutation_rate, num_params);
      break;
    case MUT_safe:
      return safe_recombine(params1, paramgrad1, params2, paramgrad2, mutation_rate, num_params);
      break;
    case MUT_safe_momentum:
      return safe_momentum_recombine(params1, paramgrad1, params2, paramgrad2, momentum, mutation_rate, num_params);
      break;
  }
  return NULL;
}

MLP *MLP_recombine(Mutator m, MLP *a, MLP *b){
  assert(a->num_params == b->num_params);
  
  MLP *ret = copy_mlp(a);
  free(ret->params);
  ret->params = recombine(m.mutation_type, a->params, a->param_grad, b->params, b->param_grad, m.momentum, m.mutation_rate, a->num_params);

  return ret;
}

RNN *RNN_recombine(Mutator m, RNN *a, RNN *b){
  assert(a->num_params == b->num_params);

  RNN *ret = copy_rnn(a);
  free(ret->params);
  ret->params = recombine(m.mutation_type, a->params, a->param_grad, b->params, b->param_grad, m.momentum, m.mutation_rate, a->num_params);

  return ret;
}

LSTM *LSTM_recombine(Mutator m, LSTM *a, LSTM *b){
  assert(a->num_params == b->num_params);

  LSTM *ret = copy_lstm(a);
  free(ret->params);
  ret->params = recombine(m.mutation_type, a->params, a->param_grad, b->params, b->param_grad, m.momentum, m.mutation_rate, a->num_params);

  return ret;
}

Mutator create_mutator(Network_type net, Mutation_type mut){
  Mutator m;
  m.network_type = net;
  m.mutation_type = mut;

  m.mutation_rate = 0.01;
  m.momentum = NULL;

  switch(net){
    case mlp:
      m.recombine = MLP_recombine;
      break;
    case rnn:
      m.recombine = RNN_recombine;
      break;
    case lstm:
      m.recombine = LSTM_recombine;
      break;
  }

}
