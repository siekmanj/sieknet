#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <locale.h>
#include <string.h>
#include <locale.h>
#include <time.h>

#include <conf.h>
#include <mlp.h>
#include <rnn.h>
#include <lstm.h>
#include <env.h>
#include <rs.h>

#ifdef NUM_THREADS
#include <omp.h>
#endif


#if !defined(USE_MLP) && !defined(USE_RNN) && !defined(USE_LSTM)
#define USE_MLP
#endif

#if defined(USE_MLP)
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
#define DIRECTIONS 100
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

#ifndef HIDDEN_LAYER_SIZE
#define HIDDEN_LAYER_SIZE 32
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
#define MAX_TRAJ_LEN 1000
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

float evaluate(Environment *env, NETWORK_TYPE *n, Normalizer *normalizer, int render, size_t *timesteps){
  float perf = 0;
  if(!render)
    env->close(*env);

  for(int i = 0; i < ROLLOUTS_PER_MEMBER; i++){
    env->reset(*env);
    env->seed(*env);

    for(int t = 0; t < MAX_TRAJ_LEN; t++){
      if(timesteps)
        *timesteps = *timesteps + 1;
      //if(timesteps && *timesteps > 2000){
        //printf("before:\n");
        //PRINTLIST(env->state, env->observation_space);
      //}
      normalize(normalizer, env);
      //if(timesteps && *timesteps > 2000){
        //printf("after:\n");
        //PRINTLIST(env->state, env->observation_space);
      //  getchar();
      //}
      forward(network_type)(n, env->state);
      perf += env->step(*env, n->output);

      if(render)
        env->render(*env);

      if(*env->done)
        break;
    }
  }
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
size_t THREAD_SAMPLES[NUM_THREADS];

/* 
 * The function pointer passed in to the random search
 * algorithm. Supports multithreading with OpenMP.
 */
float R(const float *theta, size_t len, Normalizer *normalizer){
  #ifdef _OPENMP
  size_t num_t = omp_get_thread_num();
  #else
  size_t num_t = 0;
  #endif

  Environment *env = &ENVS[num_t];
  NETWORK_TYPE *policy = &POLICIES[num_t];
  size_t *samples = &THREAD_SAMPLES[num_t];

  memcpy(policy->params, theta, len * sizeof(float));
  return evaluate(env, policy, normalizer, 0, samples);
}

size_t num_samples(){
  size_t samples = 0;
  for(int i = 0; i < NUM_THREADS; i++)
    samples += THREAD_SAMPLES[i];
  return samples;
}

char *create_logfile_name(size_t hidden_size, size_t random_seed){
  char *ret = malloc(1000*sizeof(char));
  snprintf(ret, 50, "%s", "./log/");
  snprintf(ret + strlen(ret), 50, "%u.", ALGO);
  snprintf(ret + strlen(ret), 50, "%d.", DIRECTIONS);
  snprintf(ret + strlen(ret), 50, "%s.", MACROVAL(network_type));
#ifdef USE_LINEAR
  snprintf(ret + strlen(ret), 50, "linear.");
#endif
  snprintf(ret + strlen(ret), 50, "%s.", MACROVAL(ENV_NAME));
  snprintf(ret + strlen(ret), 50, "hs.%lu.", hidden_size);
  snprintf(ret + strlen(ret), 50, "std.%3.2f.", NOISE_STD);
  snprintf(ret + strlen(ret), 50, "lr.%5.4f.", STEP_SIZE);
  snprintf(ret + strlen(ret), 50, "seed.%lu.", random_seed);
  if(NUM_THREADS > 1)
    snprintf(ret + strlen(ret), 50, "nd");
  return ret;
}


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
  char *modelfile  = argv[2];
  char *normalfile = (char*)malloc(sizeof(char) * (strlen(modelfile) + 6));
  strcpy(normalfile, modelfile);
  strcat(normalfile, ".norm");

  float default_alive_bonus;
  for(int i = 0; i < NUM_THREADS; i++){
    ENVS[i] = create_env(ENV_NAME)();
    default_alive_bonus = ENVS[i].alive_bonus;
    ENVS[i].alive_bonus = 0;
  }

  NETWORK_TYPE policy;
  Normalizer *normalizer = NULL;

  /* Load a policy from a file or create a new policy */
  if(!strcmp(argv[1], "load")){
    policy = load_policy(modelfile);
    normalizer = load_normalizer(normalfile);
    if(!normalizer)
      printf("Couldn't find '%s' - not normalizing states.\n", normalfile);
  }
  else if(!strcmp(argv[1], "new")){
    policy = new_policy(modelfile, ENVS[0].observation_space, ENVS[0].action_space);

    #ifdef USE_LINEAR
      #if (defined(USE_MLP) || defined(USE_RNN))
      for(int i = 0; i < policy.depth; i++)
        policy.layers[i].logistic = linear;
      #endif

      #if defined(USE_RNN) || defined(USE_LSTM)
      policy.output_layer.logistic = linear;
      #endif

      for(int j = 0; j < policy.num_params; j++)
        policy.params[j] = 0;

    #else
      #if (defined(USE_MLP) || defined(USE_RNN))
      for(int i = 0; i < policy.depth; i++)
        policy.layers[i].logistic = relu;
      #endif

      #if defined(USE_RNN)
      policy.output_layer.logistic = hypertan;
      #endif

      #if defined(USE_LSTM)
      policy.output_layer.logistic = hypertan;
      #endif

    #endif
  }

  printf("%s has %'lu params.\n", MACROVAL(network_type), policy.num_params);

  /* If we're evaluating a policy */
  if(!strcmp(argv[3], "eval")){
    ENVS[0].alive_bonus = default_alive_bonus;
    if(normalizer)
      normalizer->update = 0;
    while(1){
      printf("Average return over %d rollouts: %f\n", ROLLOUTS_PER_MEMBER, evaluate(&ENVS[0], &policy, normalizer, 1, NULL));
    }
  }

  /* If we're training a policy */
  else if(!strcmp(argv[3], "train")){

    /* Initialize the policy for each thread */
    for(int i = 0; i < NUM_THREADS; i++){
      POLICIES[i] = *copy(network_type)(&policy);
    }

    #ifdef _OPENMP
    printf("OpenMP detected! Using multithreading (%d threads)\n", NUM_THREADS);
    #endif

    size_t seed = RANDOM_SEED;
    srand(seed);

    char *logfile = create_logfile_name(policy.layers[0].size, seed);
    printf("Logging to '%s'\n", logfile);

    FILE *log = fopen(logfile, "wb");
    if(!log){
      printf("ERROR: main(): Couldn't open '%s' for write.\n", logfile);
      exit(1);
    }
    fprintf(log, "%s %s %s\n", "iteration", "samples", "return");

    RS r = create_rs(R, policy.params, policy.num_params, DIRECTIONS);
    r.cutoff      = TOP_B;
    r.step_size   = STEP_SIZE;
    r.std         = NOISE_STD;
    r.algo        = ALGO;
    r.num_threads = NUM_THREADS;

    if(!normalizer && (r.algo == V2 || r.algo == V2_t))
      r.normalizer = create_normalizer(policy.input_dimension);
    else
      r.normalizer = normalizer;
    
    float avg_return = 0;
    size_t iter      = 0;
    const size_t print_every = 10;

    while(num_samples() < TIMESTEPS){
			if(!(iter % print_every)){
				avg_return = 0;
				printf("\n");
			}
      size_t samples_before = num_samples();

      double start = get_time();

      rs_step(r);

      ENVS[0].alive_bonus = default_alive_bonus;
      float iter_return = evaluate(&ENVS[0], &policy, r.normalizer, 0, NULL);
      ENVS[0].alive_bonus = 0;

      size_t samples_after = num_samples();

      float samples_per_sec = (get_time() - start)/(samples_after - samples_before);
      float completion      = (double)samples_after / (double)TIMESTEPS;
      float time_left       = ((1 - completion) * TIMESTEPS) * samples_per_sec;

      int hrs_left = (int)(time_left / (60 * 60));
      int min_left = ((int)(time_left - (hrs_left * 60 * 60))) / 60;

      avg_return += iter_return;

      printf("iteration %3lu | avg return over last %2lu iters: %9.2f | time remaining %3dh %2dm | %5.4fs / 1k samples | samples %'9lu \r", iter+1, (iter % print_every) + 1, avg_return / (((iter) % print_every)+1), hrs_left, min_left, samples_per_sec * 1000, samples_after);
      fprintf(log, "%lu %lu %f\n", iter+1, samples_after, iter_return);

      save(network_type)(&policy, modelfile);
      if(r.normalizer)
	      save_normalizer(r.normalizer, normalfile);

      iter++;
    }
    fclose(log);
  }
  return 0;
}
