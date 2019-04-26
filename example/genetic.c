#include <stdlib.h>
#include <stdio.h>
#include <lstm.h>
#include <rnn.h>
#include <ga.h>
#include <env.h>
#include <hopper2d_env.h>

//#define USE_MLP
//#define USE_RNN
#define USE_LSTM

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

#define POOL_SIZE 5

#define OBS_SPACE 1
#define ACT_SPACE 1

#define HIDDEN_SIZE 1

#define ELITE_PERCENTILE 0.5f;

/*
 * Env:
 *  create()
 *  reset()
 *  step()
 *  dispose()
 *  action_space
 *  observation_space
 */


int main(){
  Environment env = create_hopper2d_env();
	srand(2);
  while(1){
    for(int i = 0; i < 100; i++){
      env.step(env, NULL);
      env.render(env);
    }
    env.close(env);
    getchar();
  }

  /*
	NETWORK_TYPE seed = create(network_type)(OBS_SPACE, HIDDEN_SIZE, ACT_SPACE);
	Pool p = create_pool(network_type, &seed, POOL_SIZE);
	p.mutation_type = MUT_baseline;
	p.mutation_rate = 0.05;
	p.elite_percentile = 0.9;

	for(int i = 0; i < p.pool_size; i++){
    float x = 0.5;
		NETWORK_TYPE *n = p.members[i];
		forward(network_type)(n, &x);
		n->performance = (float)rand()/RAND_MAX;
	}
  evolve_pool(&p);
  */
  return 0;
}
