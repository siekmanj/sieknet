#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <lstm.h>
#include <rnn.h>
#include <ga.h>
#include <env.h>
#include <hopper2d_env.h>


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
#define POOL_SIZE        1000
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

#ifndef HIDDEN_LAYER_SIZE
#define HIDDEN_LAYER_SIZE      10
#endif

#ifndef STEP_SIZE
#define STEP_SIZE        0.05f
#endif

#ifndef MUTATION_RATE
#define MUTATION_RATE    0.01f
#endif

#ifndef ELITE_PERCENTILE
#define ELITE_PERCENTILE 0.750f
#endif

#ifndef GENERATIONS
#define GENERATIONS  350
#endif 

#ifndef MAX_TRAJ_LEN
#define MAX_TRAJ_LEN 300
#endif

#ifndef RENDER_EVERY
#define RENDER_EVERY 10
#endif

#define ALGO MUT_baseline

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

#define MACROVAL_(s) #s
#define MACROVAL(s) MACROVAL_(s)

#define LOGFILE_ ./log/pool_size.POOL_SIZE.hidden_size.HIDDEN_LAYER_SIZE.step_size.STEP_SIZE.mutation_rate.MUTATION_RATE.network_type.algo.ALGO.log

int main(int argc, char** argv){
  if(argc < 3){ printf("%d args needed. Usage: [new/load] [path_to_modelfile]\n", 2); exit(1);}

	srand(2);
  setbuf(stdout, NULL);
  FILE *log = fopen(MACROVAL(LOGFILE_), "wb");

  Environment env = create_hopper2d_env();

  int newmodel;
  if(!strcmp(argv[1], "load")) newmodel = 0;
  else if(!strcmp(argv[1], "new")) newmodel = 1;
  else throw_err("nope");

  char *modelfile = argv[2];

  NETWORK_TYPE n;
  if(newmodel){
    printf("creating '%s'\n", modelfile);
    size_t layersizes[LAYERS];
    layersizes[0] = env.observation_space;
    for(int i = 1; i < LAYERS-1; i++)
      layersizes[i] = HIDDEN_LAYER_SIZE;
    layersizes[LAYERS-1] = env.action_space;

    n = from_arr(network_type)(layersizes, LAYERS);
  }else{
    printf("loading '%s'\n", modelfile);
    FILE *fp = fopen(modelfile, "rb");
    if(!fp){ printf("Could not open modelfile '%s' - does it exist?\n", modelfile); exit(1);}
    n = load(network_type)(modelfile);
    fclose(fp);
  }
  printf("network has %lu params.\n", n.num_params);


#if defined(USE_LSTM) || defined(USE_RNN)
  n.output_layer.logistic = hypertan;
#else
  n.layers[n.depth-1].logistic = hypertan;
#endif

	Pool p = create_pool(network_type, &n, POOL_SIZE);
  p.step_size = STEP_SIZE;
	p.mutation_type = ALGO;
	p.mutation_rate = MUTATION_RATE;
	p.elite_percentile = ELITE_PERCENTILE;

  for(int gen = 0; gen < GENERATIONS; gen++){
    for(int i = 0; i < p.pool_size; i++){
      NETWORK_TYPE *n = p.members[i];
      #if defined(USE_LSTM) || defined(USE_RNN)
      wipe(network_type)(n);
      #endif
      n->performance = 0;

      env.reset(env);
      env.seed(env);

      for(int t = 0; t < MAX_TRAJ_LEN; t++){

        forward(network_type)(n, env.state);
        n->performance += env.step(env, n->output);
        if(!i && gen && !(gen % RENDER_EVERY))
          env.render(env);
        if(*env.done){
          break;
        }
      }
      if(!i && gen && !(gen % RENDER_EVERY))
        env.close(env);
    }
    evolve_pool(&p);
    printf("%s %3d %6.4f\n", MACROVAL(LOGFILE_), gen, ((NETWORK_TYPE*)p.members[0])->performance);
    fprintf(log, "%f\n", ((NETWORK_TYPE*)p.members[0])->performance);
    fflush(log);
    fflush(stdout);
    save(network_type)(((NETWORK_TYPE*)p.members[0]), "./model/hopper.lstm");

  }
  fclose(log);
  return 0;
}
