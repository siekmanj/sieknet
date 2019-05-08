#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <locale.h>

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

#ifndef STEP_SIZE
#define STEP_SIZE 0.05f
#endif

#ifndef MUTATION_RATE
#define MUTATION_RATE 0.05f
#endif

#ifndef ELITE_PERCENTILE
#define ELITE_PERCENTILE 0.90f
#endif

#ifndef GENERATIONS
#define GENERATIONS 200
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
#define ROLLOUTS_PER_MEMBER 2
#endif

#ifndef CROSSOVER
#define CROSSOVER 0
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

#define create_env_(envname) create_ ## envname ## _env
#define create_env(envname) create_env_(envname)

#define MACROVAL_(s) #s
#define MACROVAL(s) MACROVAL_(s)

#if defined(USE_LSTM) || defined(USE_RNN)
  #define sensitivity(n) sensitivity_gradient(n->cost_gradient, n->output, n->output_layer.logistic, n->output_dimension)
#else
  #define sensitivity(n) sensitivity_gradient(n->cost_gradient, n->output, n->layers[n->depth-1].logistic, n->output_dimension)
#endif

#define LOGFILE_ ./log/POOL_SIZE.ENV_NAME.hs.HIDDEN_LAYER_SIZE.lr.STEP_SIZE.mr.MUTATION_RATE.network_type.MUTATION_TYPE.log

Environment envs[NUM_THREADS];
size_t samples = 0;

float evaluate(Environment *env,/* Normalizer *norm,*/ NETWORK_TYPE *n, int render){
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

			if(MUTATION_TYPE == SAFE || MUTATION_TYPE == SAFE_MOMENTUM){
				sensitivity(n);
				abs_backward(network_type)(n);
			}
			perf += env->step(*env, n->output);

      /* Normalizing doesn't seem to help very much, so not doing it for now */
      //normalize(*norm, *env);

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
  return clock() / CLOCKS_PER_SEC;
#endif
}

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

	srand(1);
  setbuf(stdout, NULL);
  FILE *log = fopen(MACROVAL(LOGFILE_), "wb");

	for(int i = 0; i < NUM_THREADS; i++){
		envs[i] = create_env(ENV_NAME)();
	}

#ifdef _OPENMP
	printf("OpenMP detected! Using multithreading (%d threads)\n", NUM_THREADS);
	omp_set_num_threads(NUM_THREADS);
#endif

  NETWORK_TYPE seed;
  if(!strcmp(argv[1], "load")){
    printf("loading '%s'\n", modelfile);
    FILE *fp = fopen(modelfile, "rb");
    if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
    fclose(fp);

    seed = load(network_type)(modelfile);
    if(seed.input_dimension != envs[0].observation_space || seed.output_dimension != envs[0].action_space){
      printf("ERROR: Policy is not compatible with environment - mismatched observation/action space shapes.\n");
      exit(1);
    }
	}else if(!strcmp(argv[1], "new")){
    printf("creating '%s'\n", modelfile);
    size_t layersizes[LAYERS];
    layersizes[0] = envs[0].observation_space;
    for(int i = 1; i < LAYERS-1; i++)
      layersizes[i] = HIDDEN_LAYER_SIZE;
    layersizes[LAYERS-1] = envs[0].action_space;

    seed = from_arr(network_type)(layersizes, LAYERS);
	}else{
    printf("Invalid arg: '%s'\n", argv[1]);
    exit(1);
  }
  printf("network has %lu params.\n", seed.num_params);

  //Normalizer normalizer = create_normalizer(envs[0], &seed, forward(network_type), seed.output, 1000);

#if defined(USE_LSTM) || defined(USE_RNN)
  seed.output_layer.logistic = hypertan;
#else
  seed.layers[seed.depth-1].logistic = hypertan;
#endif

  if(!strcmp(argv[3], "eval"))
    while(1)
			printf("Return over %d rollouts: %f\n", ROLLOUTS_PER_MEMBER, evaluate(&envs[0], /*&normalizer,*/ &seed, 1));
	else if(!strcmp(argv[3], "train")){
		
    /* Create a pool object from the seed neural network */
		Pool p = create_pool(network_type, &seed, POOL_SIZE);

    p.crossover = CROSSOVER;
		p.step_size = STEP_SIZE;
		p.mutation_type = MUTATION_TYPE;
		p.mutation_rate = MUTATION_RATE;
		p.elite_percentile = ELITE_PERCENTILE;

		float peak_fitness = 0;
		float avg_fitness = 0; 

		int print_every = 10;

		printf("logging to '%s'\n", MACROVAL(LOGFILE_));
		for(int gen = 0; gen < GENERATIONS; gen++){
			if(!(gen % print_every)){
				peak_fitness = 0;
				avg_fitness = 0;
				printf("\n");
			}
			Member **pool = p.members;
			const size_t pool_size = p.pool_size;
			float gen_avg_fitness = 0;

			size_t samples_before = samples;
      double start = get_time();
			#ifdef _OPENMP
			//double start = omp_get_wtime();
			#pragma omp parallel for default(none) shared(pool, envs,/* normalizer*/) reduction(+: gen_avg_fitness, samples)
			#endif

			for(int i = 0; i < pool_size; i++){
				int t_num = 0;
				#ifdef _OPENMP
				t_num = omp_get_thread_num();
				#endif

				NETWORK_TYPE *n = pool[i]->network;
				pool[i]->performance = evaluate(&envs[t_num], /*&normalizer,*/ n, 0);
				gen_avg_fitness += pool[i]->performance;
			}
			float testavg = 0;
			for(int i = 0; i < pool_size; i++){
				testavg += pool[i]->performance;
			}

			evolve_pool(&p);

			peak_fitness += p.members[0]->performance;
			avg_fitness  += gen_avg_fitness / p.pool_size;
			float test_return = evaluate(&envs[0], /*&normalizer,*/ ((NETWORK_TYPE*)p.members[0]->network), !(gen % print_every));

#ifndef VISDOM_OUTPUT
			printf("gen %3d | test %5.2f | 10 gen avg peak %5.2f | avg %5.2f | %4.3fs per 1k env steps | %'lu env steps      \r", gen+1, test_return, peak_fitness / (((gen) % print_every)+1), avg_fitness / (((gen) % print_every)+1), 1000*(get_time() - start)/(samples - samples_before), samples);
#else
			printf("%s %3d %6.4f %6.4f %6.4f\n", MACROVAL(LOGFILE_), gen, p.members[0]->performance, gen_avg_fitness / p.pool_size, test_return);
#endif
			fprintf(log, "%f\n", p.members[0]->performance);
			fflush(log);
			fflush(stdout);
			save(network_type)(((NETWORK_TYPE*)p.members[0]->network), modelfile);
		}
		fclose(log);
		printf("\nFinished!\n");
	}else{
    printf("Invalid arg: '%s'\n", argv[3]);
    exit(1);
	}

  return 0;
}
