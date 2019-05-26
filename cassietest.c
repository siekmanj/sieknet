#include <cassie_env.h>
#include <stdio.h>
#include <stdlib.h>

int main(){
	Environment env = create_cassie_env();
	float *x = calloc(env.action_space, sizeof(float));
	while(1){
		printf("stepping!\n");
		env.step(env, x);
		printf("rendering!\n");
		env.render(env);
	}
	printf("Done.\n");
}
