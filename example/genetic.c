#include <stdlib.h>
#include <stdio.h>
#include <lstm.h>
#include <rnn.h>
#include <ga.h>

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

#define create_pool_(arch) create_ ## arch ## _pool
#define create_pool(arch) create_pool_(arch)

#define network_pool_(ARCH) ARCH ## _pool
#define network_pool(ARCH) network_pool_(ARCH)

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
	srand(2);
  
	/*
  NETWORK_TYPE **pool = ALLOC(NETWORK_TYPE*, POOL_SIZE);

  for(int i = 0; i < POOL_SIZE; i++){
    pool[i] = ALLOC(NETWORK_TYPE, 1);
    *pool[i] = create(network_type)(OBS_SPACE, HIDDEN_SIZE, ACT_SPACE);
    pool[i]->performance = 0.0f;
  }
	*/
	int random = 1;
	NETWORK_TYPE seed = create(network_type)(OBS_SPACE, HIDDEN_SIZE, ACT_SPACE);

	network_pool(NETWORK_TYPE) pool = create_pool(network_type)(POOL_SIZE, &seed, random);

  Mutator m = create_mutator(network_type, MUT_baseline);
	m.mutation_rate = 0.1;
	

  /*

  for(int gen = 0; gen < GENERATIONS; gen++){
    //TODO: ENV SETUP
    Environment env;
    for(int i = 0; i < POOL_SIZE; i++){
      for(int t = 0; t < MAX_TIMESTEPS; t++){

        //MUJOCO STUFF HERE

        float *obs; //TODO

        forward(network_type)(pool[i], env->state);
        set_sensitivity_gradient(pool[i]->mlp_cost_gradient, pool[i]->output, pool[i]->output_layer.logistic);
        backward(network_type)(pool[i]);

        //TODO: ENV STEP
        reward += env->step(env, action);
        

      }
    }
    //TODO: SORTING
    //qsort(pool);

    //Cull everyone but elites
    for(int i = (int)(ELITE_PERCENTILE * POOL_SIZE); i < POOL_SIZE; i++){
      dealloc(network_type)(&pool[i]);
    }
    
    //repopulate pool
    for(int i = 0; i < (int)(ELITE_PERCENTAGE * POOL_SIZE); i++){

    }



  }

  
  */
  return 0;
}
