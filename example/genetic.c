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
	NETWORK_TYPE seed = create(network_type)(OBS_SPACE, HIDDEN_SIZE, ACT_SPACE);

	Pool p = create_pool(network_type, &seed, POOL_SIZE);
	p.mutation_type = MUT_baseline;
	p.mutation_rate = 0.05;
	p.elite_percentile = 0.9;

	for(int i = 0; i < p.pool_size; i++){
		NETWORK_TYPE *n = p.members[i];
		forward(network_type)(n, NULL);
		//set_sensitivity_gradient(n->cost_gradient, n->output, n->output_layer->logistic);
		backward(network_type)(n);

		n->performance += 0.05;
	}

	sort_pool(&p);
	cull_pool(&p);
	breed_pool(&p);



/*
	network_pool(NETWORK_TYPE) pool = create_pool(network_type)(POOL_SIZE, &seed, 1);

  Mutator m = create_mutator(network_type, MUT_baseline);
	m.mutation_rate = 0.1;

  for(int i = 0; i < POOL_SIZE; i++){
    pool[i]->performance = (float)rand() / RAND_MAX;
    printf("pool[%d] has perf %f\n", i, pool[i]->performance);
  }
  sort_pool(network_type)(pool, POOL_SIZE);
  for(int i = 0; i < POOL_SIZE; i++){
    printf("pool[%d] now has perf %f\n", i, pool[i]->performance);
  }

  cull_pool(network_type)(pool, m, POOL_SIZE);
	for(int i = 0; i < POOL_SIZE; i++)
		printf("pool[%d]: %p\n", i, pool[i]);
	breed_pool(network_type)(pool, m, POOL_SIZE);
	for(int i = 0; i < POOL_SIZE; i++)
		printf("pool[%d]: %p\n", i, pool[i]);
*/
	


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
    

    }
  }
  */
  return 0;
}
