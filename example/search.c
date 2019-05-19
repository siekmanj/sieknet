#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <locale.h>
#include <string.h>
#include <locale.h>
#include <time.h>

#include <conf.h>
#include <lstm.h>
#include <rnn.h>
#include <ga.h>
#include <env.h>

#ifdef NUM_THREADS
#include <omp.h>
#endif

#include <rs.h>

#if !defined(USE_MLP) && !defined(USE_RNN) && !defined(USE_LSTM) && !defined(USE_LINEAR)
#define USE_LINEAR
#endif

#if defined(USE_LINEAR) || defined(USE_MLP)
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

#ifndef DIRECTIONS
#define DIRECTIONS 60
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

#ifndef HIDDEN_LAYER_SIZE
#define HIDDEN_LAYER_SIZE 10
#endif

#ifndef STEP_SIZE
#define STEP_SIZE 0.02f
#endif

#ifndef NOISE_STD
#define NOISE_STD 0.02f
#endif

#ifndef TOP_B
#define TOP_B 0.0f
#endif

#ifndef ALGO
#define ALGO V1
#endif

#ifndef TIMESTEPS
#define TIMESTEPS 4e8
#endif

#ifndef MAX_TRAJ_LEN
#define MAX_TRAJ_LEN 600
#endif

#ifndef ENV_NAME
#define ENV_NAME half_cheetah
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

#ifndef ROLLOUTS_PER_MEMBER
#define ROLLOUTS_PER_MEMBER 1
#endif

#ifndef RANDOM_SEED
#define RANDOM_SEED time(0)
#endif

#define MAKE_INCLUDE_(envname) <envname ## _env.h>
#define MAKE_INCLUDE(envname) MAKE_INCLUDE_(envname)

/* Don't try this at home, kids */
#include MAKE_INCLUDE(ENV_NAME)

/* Some ghetto polymorphism */

#define forward_(arch) arch ## _forward
#define forward(arch) forward_(arch)

#define create_(arch) create_ ## arch
#define create(arch) create_(arch)

#define dealloc_(arch) dealloc_ ## arch
#define dealloc(arch) dealloc_(arch)

#define wipe_(arch) arch ## _wipe
#define wipe(arch) wipe_(arch)

#define from_arr_(arch) arch ## _from_arr
#define from_arr(arch) from_arr_(arch)

#define load_(arch) load_ ## arch
#define load(arch) load_(arch)

#define save_(arch) save_ ## arch
#define save(arch) save_(arch)

#define copy_(arch) copy_ ## arch
#define copy(arch) copy_(arch)

#define create_env_(envname) create_ ## envname ## _env
#define create_env(envname) create_env_(envname)

#define MACROVAL_(s) #s
#define MACROVAL(s) MACROVAL_(s)

size_t samples = 0;

float evaluate(Environment *env, NETWORK_TYPE *n, int render){
  float perf = 0;
  for(int i = 0; i < ROLLOUTS_PER_MEMBER; i++){
    env->reset(*env);
    env->seed(*env);

    for(int t = 0; t < MAX_TRAJ_LEN; t++){
      samples++;
      mlp_forward(n, env->state);
      perf += env->step(*env, n->output);

      if(render)
        env->render(*env);

      if(*env->done)
        break;
    }
  }
  if(render)
    env->close(*env);

  return perf / ROLLOUTS_PER_MEMBER;
}

NETWORK_TYPE load_policy(char *modelfile){
  printf("loading '%s'\n", modelfile);
  FILE *fp = fopen(modelfile, "rb");
  if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
  fclose(fp);

  return load(network_type)(modelfile);
}

NETWORK_TYPE new_policy(char *modelfile, size_t obs_space, size_t act_space){
  printf("creating '%s'\n", modelfile);
  size_t layersizes[LAYERS];
  layersizes[0] = obs_space;
  for(int i = 1; i < LAYERS-1; i++)
    layersizes[i] = HIDDEN_LAYER_SIZE;
  layersizes[LAYERS-1] = act_space;

  return from_arr(network_type)(layersizes, LAYERS);
}
double get_time(){
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
}

NETWORK_TYPE POLICIES[NUM_THREADS];
Environment ENVS[NUM_THREADS];

int main(int argc, char **argv){
  if(argc < 4){ printf("%d args needed. Usage: [new/load] [path_to_modelfile] [train/eval]\n", 3); exit(1);}
  setlocale(LC_ALL,"");

  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	   \n");
  printf("																					 \n");
  printf("augmented random search for reinforcement learning.\n");

  setbuf(stdout, NULL);
  char *modelfile = argv[2];

  for(int i = 0; i < NUM_THREADS; i++){
    ENVS[i] = create_env(ENV_NAME)();
  }

  NETWORK_TYPE policy;

  /* Load a policy from a file or create a new policy */
  if(!strcmp(argv[1], "load"))
    policy = load_policy(modelfile);

  else if(!strcmp(argv[1], "new"))
    policy = new_policy(modelfile, ENVS[0].observation_space, ENVS[0].action_space);

#ifdef USE_LINEAR
  for(int i = 0; i < policy.depth; i++)
    policy.layers[i].logistic = linear;

  for(int j = 0; j < policy.num_params; j++)
    policy.params[j] = 0;
#endif

  /* Initialize the policy for each thread */
  for(int i = 0; i < NUM_THREADS; i++){
    POLICIES[i] = *copy(network_type)(&policy);
  }

  /* If we're evaluating a policy */
  if(!strcmp(argv[3], "eval"))
    while(1)
      printf("Return over %d rollouts: %f\n", ROLLOUTS_PER_MEMBER, evaluate(&ENVS[0], &policy, 1));

  /* If we're training a policy */
  else if(!strcmp(argv[3], "train")){
    #ifdef _OPENMP
    printf("OpenMP detected! Using multithreading (%d threads)\n", NUM_THREADS);
		omp_set_num_threads(NUM_THREADS);
    #endif

    srand(time(NULL));
    RS r = create_rs(policy.params, policy.num_params, DIRECTIONS);
    r.cutoff    = TOP_B;
    r.step_size = STEP_SIZE;
    r.std       = NOISE_STD;
    r.algo      = ALGO;
    
    size_t episodes = 0;
    int iter = 0;
    while(samples < TIMESTEPS){
      iter++;
      size_t samples_before = samples;
      double start = get_time();

      #ifdef _OPENMP
      #pragma omp parallel for default(none) shared(policy, r, ENVS, POLICIES) reduction(+: samples, episodes)
      #endif
      for(int i = 0; i < r.directions; i++){
        #ifdef _OPENMP
        int num_t = omp_get_thread_num();
        #else
        int num_t = 0;
        #endif

				//printf("thread %d: params %p, theta %p, deltas %p\n", num_t, POLICIES[num_t].params, theta, r.deltas[i]->p);

        for(int j = 0; j < policy.num_params; j++)
          POLICIES[num_t].params[j] = policy.params[j] + r.deltas[i]->p[j];
        r.deltas[i]->r_pos = evaluate(&ENVS[num_t], &POLICIES[num_t], 0);

        for(int j = 0; j < policy.num_params; j++)
          POLICIES[num_t].params[j] = policy.params[j] - r.deltas[i]->p[j];
        r.deltas[i]->r_neg = evaluate(&ENVS[num_t], &POLICIES[num_t], 0);
      
        episodes += 2 * ROLLOUTS_PER_MEMBER;
      }
      rs_step(r);

      float completion = (double)samples / (double)TIMESTEPS;
      float samples_per_sec = (get_time() - start)/(samples - samples_before);
      float time_left = ((1 - completion) * TIMESTEPS) * samples_per_sec;
      int hrs_left = (int)(time_left / (60*60));
      int min_left = ((int)(time_left - (hrs_left * 60 * 60))) / 60;
      printf("iteration %3d | reward: %9.2f | est. time remaining %3dh %2dm | %5.4fs per 1k samples | episodes %7lu | samples %'9lu \r", iter, evaluate(&ENVS[0], &policy, 0), hrs_left, min_left, samples_per_sec * 1000, episodes, samples);
      if(!(iter%10))
        printf("\n");

      save(network_type)(&policy, modelfile);
    }
  }
  return 0;
}
