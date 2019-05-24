#include <cassie_env.h>
#include <stdio.h>
#include <stdlib.h>

void dispose(Environment env){

}

void reset(Environment env){

}

void seed(Environment env){

}

void render(Environment env){

}

void close(Environment env){

}

float step(Environment env, float *action){

  return 0.0f;
}

Environment create_cassie_env(){
  Environment env;

  env.dispose = dispose;
  env.reset = reset;
  env.seed = seed;
  env.render = render;
  env.close = close;
  env.step = step;

  //Data *d = (Data*)malloc(sizeof(Data));

  env.data = NULL;
  env.state = NULL;

  env.observation_space = 0; //TODO
  env.action_space = 0; //TODO

  env.done = calloc(1, sizeof(int));

  return env;
}
