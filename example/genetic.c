#include <stdlib.h>
#include <stdio.h>
#include <lstm.h>
#include <rnn.h>
#include <ga.h>
#include <env.h>
#include <hopper2d_env.h>

#define USE_MLP
//#define USE_RNN
//#define USE_LSTM

#ifdef USE_MLP
#define network_type mlp
#define NETWORK_TYPE MLP
#endif

#ifdef USE_RNN
#define network_type rnn
#define NETWORK_TYPE RNN
#endif

#ifdef USE_LSTM
#define network_type lstm
#define NETWORK_TYPE LSTM
#endif

/* Some ghetto polymorphism */

#define forward_(arch) arch ## _forward
#define forward(arch) forward_(arch)

#define cost_(arch) arch ## _cost
#define cost(arch) cost_(arch)

#define backward_(arch) arch ## _backward
#define backward(arch) backward_(arch)

#define create_(arch) create_ ## arch
#define create(arch) create_(arch)

#define dealloc_(arch) dealloc_ ## arch
#define dealloc(arch) dealloc_(arch)

#define POOL_SIZE 250

#define HIDDEN_SIZE 10

#define STEP_SIZE 0.005f
#define MUTATION_RATE 0.01f
#define ELITE_PERCENTILE 0.5f

#define MAX_TRAJ_LEN 400

int main(){
	srand(2);

  Environment env = create_hopper2d_env();

	NETWORK_TYPE seed = create(network_type)(env.observation_space, HIDDEN_SIZE, env.action_space);

#if defined(USE_LSTM) || defined(USE_RNN)
  seed.output_layer.logistic = hypertan;
#else
  seed.layers[seed.depth-1].logistic = hypertan;
#endif

	Pool p = create_pool(network_type, &seed, POOL_SIZE);
  p.step_size = STEP_SIZE;
	p.mutation_type = MUT_baseline;
	p.mutation_rate = MUTATION_RATE;
	p.elite_percentile = ELITE_PERCENTILE;

  for(int gen = 0; gen < 1000; gen++){
    for(int i = 0; i < p.pool_size; i++){
      NETWORK_TYPE *n = p.members[i];
      n->performance = 0;

      env.reset(env);
      env.seed(env);

      for(int t = 0; t < MAX_TRAJ_LEN; t++){

        forward(network_type)(n, env.state);
        n->performance += env.step(env, n->output);
        if(!i && gen && !(gen % 15))
          env.render(env);
        if(*env.done){
          break;
        }
      }
      if(!i && gen && !(gen % 15))
        env.close(env);
    }
    evolve_pool(&p);
    printf("done with gen %d, best fitness %f\n", gen, ((NETWORK_TYPE*)p.members[0])->performance);
  }
  return 0;
}
