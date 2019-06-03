#include <cassie_env.h>
#include <cassiemujoco.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
	Environment env = create_cassie_env();
	float *x = calloc(env.action_space, sizeof(float));
  int do_render = 1;
  for(int i = 0; i < 10000; i++){
    env.step(env, x);

    if(do_render){
      printf("rendering!\n");
    }else{
      printf("not rendering\n");
      env.close(env);
    }

    if(i && !(i % 100))
      do_render = !do_render;

    if(do_render) 
      env.render(env);
	}
	printf("Done.\n");
}
