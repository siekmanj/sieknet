#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <env.h>
#include <conf.h>

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(0, 1);
	float u2 = uniform(0, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * 3.14159 * u2);
	return mean + norm * std;
}

Normalizer create_normalizer(Environment env, void *policy, void (*forward)(void *policy, float *input), float *output, size_t samples){
  Normalizer n;
  n.dimension = env.observation_space;
  n.env_mean = calloc(n.dimension, sizeof(float));
  n.env_std = calloc(n.dimension, sizeof(float));

  float tmp_var[env.observation_space];
  memset(tmp_var, '\0', n.dimension * sizeof(float));
  
  float **raw = ALLOC(float*, samples);
  for(int t = 0; t < samples; t++)
    raw[t] = ALLOC(float, n.dimension);

  env.reset(env);
  env.seed(env);
  for(int t = 0; t < samples; t++){
    /* Generate an action, assume hypertan [-1,1] is used */
    /*
    float action[env.action_space];
    for(int j = 0; j < env.action_space; j++)
      action[j] = normal(0, .5);
    */
    forward(policy, env.state);
    
    /* Take an environment step */
    env.step(env, output);
    for(int i = 0; i < n.dimension; i++){
      raw[t][i] = env.state[i];
      n.env_mean[i] += raw[t][i];
    }

    if(*env.done){
      env.reset(env);
      env.seed(env);
    }
  }

  for(int i = 0; i < n.dimension; i++){
    n.env_mean[i] /= samples;
  }
  
  for(int t = 0; t < samples; t++){
    for(int i = 0; i < n.dimension; i++){
      tmp_var[i] += (n.env_mean[i] - raw[t][i])*(n.env_mean[i] - raw[t][i]);
    }
  }

  for(int i = 0; i < n.dimension; i++){
    tmp_var[i] /= samples-1;
    //n.env_std[i] = sqrt(tmp_var[i]);
    n.env_std[i] = 1;
    n.env_mean[i] = 0;
  }

  /*
  PRINTLIST(n.env_mean, n.dimension);
  PRINTLIST(tmp_var, n.dimension);
  PRINTLIST(n.env_std, n.dimension);
  getchar();
  */

  for(int t = 0; t < samples; t++)
    free(raw[t]);
  free(raw);
  return n;
}

void normalize(Normalizer n, Environment e){
  for(int i = 0; i < n.dimension; i++){
    e.state[i] = (e.state[i] - n.env_mean[i]) / n.env_std[i];
  }
}
