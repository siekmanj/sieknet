#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lstm.h>
#include <rnn.h>
#include <ga.h>
#include <env.h>

#include <omp.h>

#if !defined(USE_MLP) && !defined(USE_RNN) && !defined(USE_LSTM)
#define USE_LSTM
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
#define POOL_SIZE 1000
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
#define MUTATION_RATE 0.01f
#endif

#ifndef ELITE_PERCENTILE
#define ELITE_PERCENTILE 0.95f
#endif

#ifndef GENERATIONS
#define GENERATIONS 350
#endif 

#ifndef MAX_TRAJ_LEN
#define MAX_TRAJ_LEN 300
#endif

#ifndef RENDER_EVERY
#define RENDER_EVERY 10
#endif

#ifndef MUTATION_TYPE
#define MUTATION_TYPE BASELINE
#endif

#ifndef ENV_NAME
#define ENV_NAME hopper
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ROLLOUTS_PER_MEMBER
#define ROLLOUTS_PER_MEMBER 3
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

#define LOGFILE_ ./log/POOL_SIZE.ENV_NAME.hidden_size.HIDDEN_LAYER_SIZE.step_size.STEP_SIZE.mutation_rate.MUTATION_RATE.network_type.MUTATION_TYPE.log

Environment envs[NUM_THREADS];

int main(int argc, char** argv){
  if(argc < 4){ printf("%d args needed. Usage: [new/load] [path_to_modelfile] [train/eval]\n", 3); exit(1);}

  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	   \n");
  printf("																					 \n");
  printf("genetic algorithms for reinforcement learning.\n");

	srand(2);
  setbuf(stdout, NULL);
  FILE *log = fopen(MACROVAL(LOGFILE_), "wb");

	for(int i = 0; i < NUM_THREADS; i++){
		envs[i] = create_env(ENV_NAME)();
	}

#ifdef _OPENMP
	printf("OpenMP detected! Using multithreading.\n");
	omp_set_num_threads(NUM_THREADS);
#endif

  int newmodel;
  if(!strcmp(argv[1], "load")) newmodel = 0;
  else if(!strcmp(argv[1], "new")) newmodel = 1;
  else{
    printf("Invalid arg: '%s'\n", argv[1]);
    exit(1);
  }
  char *modelfile = argv[2];

  int eval;
  if(!strcmp(argv[3], "train")) eval = 0;
  else if(!strcmp(argv[3], "eval")) eval = 1;
  else{
    printf("Invalid arg: '%s'\n", argv[3]);
    exit(1);
  }

  NETWORK_TYPE seed;
  if(newmodel){
    printf("creating '%s'\n", modelfile);
    size_t layersizes[LAYERS];
    layersizes[0] = envs[0].observation_space;
    for(int i = 1; i < LAYERS-1; i++)
      layersizes[i] = HIDDEN_LAYER_SIZE;
    layersizes[LAYERS-1] = envs[0].action_space;

    seed = from_arr(network_type)(layersizes, LAYERS);
  }else{
    printf("loading '%s'\n", modelfile);
    FILE *fp = fopen(modelfile, "rb");
    if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
    seed = load(network_type)(modelfile);
    fclose(fp);
    if(seed.input_dimension != envs[0].observation_space || seed.output_dimension != envs[0].action_space){
      printf("ERROR: Policy is not compatible with environment - mismatched observation/action space shapes.\n");
      exit(1);
    }
  }
  printf("network has %lu params.\n", seed.num_params);

#if defined(USE_LSTM) || defined(USE_RNN)
  seed.output_layer.logistic = hypertan;
  #define sensitivity(n) sensitivity_gradient(n->cost_gradient, n->output, n->output_layer.logistic, n->output_dimension)
#else
  seed.layers[seed.depth-1].logistic = hypertan;
  #define sensitivity(n) sensitivity_gradient(n->cost_gradient, n->output, n->layers[n->depth-1].logistic, n->output_dimension)
#endif

  if(eval){
    while(1){
      envs[0].reset(envs[0]);
      envs[0].seed(envs[0]);
      seed.performance = 0;
      for(int t = 0; t < MAX_TRAJ_LEN; t++){
        forward(network_type)(&seed, envs[0].state);
        seed.performance += envs[0].step(envs[0], seed.output);
        envs[0].render(envs[0]);
        if(*envs[0].done){
          break;
        }
      }
      printf("Return: %f\n", seed.performance);
    }
    exit(0);
  }


	Pool p = create_pool(network_type, &seed, POOL_SIZE);
  p.step_size = STEP_SIZE;
	p.mutation_type = MUTATION_TYPE;
	p.mutation_rate = MUTATION_RATE;
	p.elite_percentile = ELITE_PERCENTILE;

	float peak_fitness = 0;
	float avg_fitness = 0; 
	size_t samples = 0;

	int print_every = 10;

  printf("logging to '%s'\n", MACROVAL(LOGFILE_));
  for(int gen = 0; gen < GENERATIONS; gen++){
		float gen_avg_fitness = 0;
		#ifdef _OPENMP
		double start = omp_get_wtime();
		#pragma omp parallel for default(none) shared(p, envs) reduction(+: gen_avg_fitness, samples)
		#endif
    for(int i = 0; i < p.pool_size; i++){
      NETWORK_TYPE *n = p.members[i];
      n->performance = 0;

			int t_num = 0;
			#ifdef _OPENMP
			t_num = omp_get_thread_num();
			#endif
			/* best of three */ 
			for(int try = 0; try < ROLLOUTS_PER_MEMBER; try++){
				#if defined(USE_LSTM) || defined(USE_RNN)
				wipe(network_type)(n);
				n->seq_len = MAX_TRAJ_LEN < SIEKNET_MAX_UNROLL_LENGTH ? MAX_TRAJ_LEN : SIEKNET_MAX_UNROLL_LENGTH;
				#endif
				envs[t_num].reset(envs[t_num]);
				envs[t_num].seed(envs[t_num]);

				for(int t = 0; t < MAX_TRAJ_LEN; t++){
					forward(network_type)(n, envs[t_num].state);

          if(MUTATION_TYPE == SAFE || MUTATION_TYPE == SAFE_MOMENTUM){
            sensitivity(n);
            backward(network_type)(n);
          }

					n->performance += envs[t_num].step(envs[t_num], n->output);
					if(*envs[0].done){
						samples += t;
						break;
					}
				}
				if(!*envs[0].done)
					samples += MAX_TRAJ_LEN;
			}
			n->performance /= ROLLOUTS_PER_MEMBER;
			gen_avg_fitness += n->performance;
    }
    evolve_pool(&p);

#ifndef VISDOM_OUTPUT
		if(gen && !(gen % print_every)){
			peak_fitness = 0;
			avg_fitness = 0;
			printf("\n");
		}
		peak_fitness += ((NETWORK_TYPE*)p.members[0])->performance;
		avg_fitness += gen_avg_fitness / p.pool_size;
		printf("gen %3d | avg peak %5.2f | avg %5.2f | %4.3fs per gen | samples %lu      \r", gen+1, peak_fitness / (((gen) % print_every)+1), avg_fitness / (((gen) % print_every)+1), omp_get_wtime() - start, samples);
#else
    printf("%s %3d %6.4f, %6.4f\n", MACROVAL(LOGFILE_), gen, ((NETWORK_TYPE*)p.members[0])->performance, gen_avg_fitness / p.pool_size);
#endif
    fprintf(log, "%f\n", ((NETWORK_TYPE*)p.members[0])->performance);
    fflush(log);
    fflush(stdout);
    save(network_type)(((NETWORK_TYPE*)p.members[0]), modelfile);
  }
  fclose(log);
  printf("\nFinished!\n");
  return 0;
}
