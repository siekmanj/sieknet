#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <locale.h>
#include <time.h>

#include <lstm.h>
#include <rnn.h>
#include <ga.h>
#include <env.h>

#include <omp.h>

#if !defined(USE_MLP) && !defined(USE_RNN) && !defined(USE_LSTM)
#define USE_MLP
#endif

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

#ifndef POOL_SIZE
#define POOL_SIZE 100
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

#ifndef HIDDEN_LAYER_SIZE
#define HIDDEN_LAYER_SIZE 10
#endif

#ifndef NOISE_STD
#define NOISE_STD 0.5f
#endif

#ifndef MUTATION_RATE
#define MUTATION_RATE 0.05f
#endif

#ifndef ELITE_PERCENTILE
#define ELITE_PERCENTILE 0.90f
#endif

#ifndef TIMESTEPS
#define TIMESTEPS 4e6
#endif

#ifndef MAX_TRAJ_LEN
#define MAX_TRAJ_LEN 400
#endif

#ifndef MUTATION_TYPE
#define MUTATION_TYPE MOMENTUM
#endif

#ifndef ENV_NAME
#define ENV_NAME hopper
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

#ifndef ROLLOUTS_PER_MEMBER
#define ROLLOUTS_PER_MEMBER 3
#endif

#ifndef CROSSOVER
#define CROSSOVER 0
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

#define cost_(arch) arch ## _cost
#define cost(arch) cost_(arch)

#define backward_(arch) arch ## _backward
#define backward(arch) backward_(arch)

#define abs_backward_(arch) arch ## _abs_backward
#define abs_backward(arch) abs_backward_(arch)

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

#if defined(USE_LSTM) || defined(USE_RNN)
  #define sensitivity(n) sensitivity_gradient(n->cost_gradient, n->output, n->output_layer.logistic, n->output_dimension)
#else
  #define sensitivity(n) sensitivity_gradient(n->cost_gradient, n->output, n->layers[n->depth-1].logistic, n->output_dimension)
#endif


size_t samples = 0;

float evaluate(Environment *env, NETWORK_TYPE *n, int render){
	float perf = 0;
	for(int i = 0; i < ROLLOUTS_PER_MEMBER; i++){
		env->reset(*env);
		env->seed(*env);

		#if defined(USE_LSTM) || defined(USE_RNN)
		wipe(network_type)(n);
		n->seq_len = MAX_TRAJ_LEN < SIEKNET_MAX_UNROLL_LENGTH ? MAX_TRAJ_LEN : SIEKNET_MAX_UNROLL_LENGTH;
		#endif

		for(int t = 0; t < MAX_TRAJ_LEN; t++){
			samples++;
			forward(network_type)(n, env->state);

			if(MUTATION_TYPE == SAFE || MUTATION_TYPE == SAFE_MOMENTUM || MUTATION_TYPE == AGGRESSIVE){
				sensitivity(n);
				abs_backward(network_type)(n);
			}
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

double get_time(){
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
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

char *create_logfile_name(size_t hidden_size, size_t random_seed){
  char *ret = malloc(1000*sizeof(char));
  snprintf(ret, 50, "%s", "./log/");
  snprintf(ret + strlen(ret), 50, "%d.", POOL_SIZE);
  snprintf(ret + strlen(ret), 50, "%s.", MACROVAL(network_type));
  snprintf(ret + strlen(ret), 50, "%s.", MACROVAL(ENV_NAME));
  snprintf(ret + strlen(ret), 50, "hs.%lu.", hidden_size);
  snprintf(ret + strlen(ret), 50, "std.%3.2f.", NOISE_STD);
  snprintf(ret + strlen(ret), 50, "mr.%3.2f.", MUTATION_RATE);
  snprintf(ret + strlen(ret), 50, "co.%d.", CROSSOVER);
  snprintf(ret + strlen(ret), 50, "seed.%lu.", random_seed);
  if(NUM_THREADS > 1)
    snprintf(ret + strlen(ret), 50, "nd");
  return ret;

}

Environment ENVS[NUM_THREADS];
NETWORK_TYPE POLICIES[NUM_THREADS];

int main(int argc, char** argv){
  if(argc < 4){ printf("%d args needed. Usage: [new/load] [path_to_modelfile] [train/eval]\n", 3); exit(1);}
  setlocale(LC_ALL,"");

  char *modelfile = argv[2];

  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	   \n");
  printf("																					 \n");
  printf("genetic algorithms for reinforcement learning.\n");

  setbuf(stdout, NULL);

	for(int i = 0; i < NUM_THREADS; i++){
		ENVS[i] = create_env(ENV_NAME)();
	}
	srand(RANDOM_SEED);

#ifdef _OPENMP
	printf("OpenMP detected! Using multithreading (%d threads)\n", NUM_THREADS);
	omp_set_num_threads(NUM_THREADS);
#endif

  /* Load a policy from a file or create a new policy */
  NETWORK_TYPE seed;
  if(!strcmp(argv[1], "load"))
    seed = load_policy(modelfile);

	else if(!strcmp(argv[1], "new"))
    seed = new_policy(modelfile, ENVS[0].observation_space, ENVS[0].action_space);
	
  else{ printf("Invalid arg: '%s'\n", argv[1]); exit(1); }

  /* Make sure that the policy is compatible with the environment */
  if(seed.input_dimension != ENVS[0].observation_space || seed.output_dimension != ENVS[0].action_space){
    printf("ERROR: Policy '%s' is not compatible with environment '%s'.\n", modelfile, MACROVAL(ENV_NAME));
    exit(1);
  }
  size_t rand_seed = RANDOM_SEED;

  printf("network has %'lu params.\n", seed.num_params);

  /* Probably want full dynamic range (-1, 1), set it just in case */
#if defined(USE_LSTM) || defined(USE_RNN)
  seed.output_layer.logistic = hypertan;
#else
  seed.layers[seed.depth-1].logistic = hypertan;
#endif

  /* Copy policies so can use a fresh one in every thread */
  for(int i = 0; i < NUM_THREADS; i++){
    POLICIES[i] = *copy(network_type)(&seed);

    free(POLICIES[i].params);     /* We don't need each policy to have a parameter vector - that's inserted by the algo */
    free(POLICIES[i].param_grad); /* Likewise don't need a parameter gradient, also inserted by algo */
    POLICIES[i].params = NULL;
    POLICIES[i].param_grad = NULL;
  }

  //evaluate(&ENVS[0], &seed, 0);
  /* If we're evaluating a policy */
  if(!strcmp(argv[3], "eval"))
    while(1)
			printf("Return over %d rollouts: %f\n", ROLLOUTS_PER_MEMBER, evaluate(&ENVS[0], &seed, 1));

  /* If we're training a policy */
	else if(!strcmp(argv[3], "train")){

    /* Create a logfile */
    char *logfile = create_logfile_name(seed.layers[0].size, rand_seed);
    printf("logging to '%s'\n", logfile);
    FILE *log = fopen(logfile, "wb");
    if(!log) { printf("unable to open '%s' for write - aborting\n", logfile); exit(1); }
		
    /* Create a GA pool object from the seed neural network */
		GA p = create_ga(NULL, seed.num_params, POOL_SIZE);

    p.crossover = CROSSOVER;
		p.noise_std = NOISE_STD;
		p.mutation_type = MUTATION_TYPE;
		p.mutation_rate = MUTATION_RATE;
		p.elite_percentile = ELITE_PERCENTILE;

		float peak_fitness = 0;
		float avg_fitness = 0; 

		int print_every = 10;

		fprintf(log, "%s %s %s %s\n", "gen", "samples", "fitness", "avgfitness");
		int gen = 0;
		while(samples < TIMESTEPS){
			if(!(gen % print_every)){
				peak_fitness = 0;
				avg_fitness = 0;
				printf("\n");
			}
			Member **pool = p.members;
			const size_t pool_size = p.size;
			float gen_avg_fitness = 0;

      #ifndef VISDOM_OUTPUT
			size_t samples_before = samples;
      double start = get_time();
      #endif
			#ifdef _OPENMP
			#pragma omp parallel for default(none) shared(pool, ENVS, POLICIES) reduction(+: gen_avg_fitness, samples)
			#endif

			for(int i = 0; i < pool_size; i++){
				int t_num = 0;
				#ifdef _OPENMP
				t_num = omp_get_thread_num();
				#endif

        NETWORK_TYPE *n  = &POLICIES[t_num];
        Environment *env = &ENVS[t_num];

        n->params = pool[i]->params;
        n->param_grad = pool[i]->param_grad;

				pool[i]->performance = evaluate(env, n, 0);
				gen_avg_fitness += pool[i]->performance;
			}
			ga_evolve(&p);

			float test_return;
      {
        NETWORK_TYPE *n = &POLICIES[0];
        n->params = p.members[0]->params;
        save(network_type)(n, modelfile);
        test_return = evaluate(&ENVS[0], n, 0);
      }

			peak_fitness += p.members[0]->performance;
			avg_fitness  += gen_avg_fitness / p.size;


#ifndef VISDOM_OUTPUT
      float completion = (double)samples / (double)TIMESTEPS;
      float samples_per_sec = (get_time() - start)/(samples - samples_before);
      float time_left = ((1 - completion) * TIMESTEPS) * samples_per_sec;
      int hrs_left = (int)(time_left / (60*60));
      int min_left = ((int)(time_left - (hrs_left * 60 * 60))) / 60;
			printf("gen %3d | test %6.2f | %2d gen avg peak %6.2f | avg %6.2f | %4.3fs per 1k env steps | est. %3dh %2dm left | %'9lu env steps      \r", gen+1, test_return, (gen % print_every)+1, peak_fitness / (((gen) % print_every)+1), avg_fitness / (((gen) % print_every)+1), 1000*samples_per_sec, hrs_left, min_left, samples);
#else
			printf("%s %3d %6lu %6.4f %6.4f %6.4f\n", MACROVAL(LOGFILE_), gen, samples, p.members[0]->performance, gen_avg_fitness / p.size, test_return);
#endif
			fprintf(log, "%d %lu %f %f\n", gen, samples, p.members[0]->performance, gen_avg_fitness / p.size);
			fflush(log);
			fflush(stdout);
			gen++;
		}
		fclose(log);
		printf("\nFinished!\n");
	}else{
    printf("Invalid arg: '%s'\n", argv[3]);
    exit(1);
	}
  return 0;
}
