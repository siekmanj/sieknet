#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H


/*
 * Use this struct to interface an environment.
 *
 */
typedef struct env_ {
  float *state;
  void *data;
  
  size_t action_space;
  size_t observation_space;

  int *done;

  void (*dispose)(struct env_ env);
  void (*reset)(struct env_ env);
  void (*seed)(struct env_ env);
  void (*render)(struct env_ env);
  void (*close)(struct env_ env);
  float (*step)(struct env_ env, float *action);
  
} Environment;

typedef struct normalize_ {
  float *env_mean;
  float *env_std;
  size_t dimension;
} Normalizer;

Normalizer create_normalizer(Environment env, void *policy, void (*forward)(void *policy, float *input), float *output, size_t samples);

void normalize(Normalizer, Environment);


#endif
