#include <conf.h>
#include <optimizer.h>
#include <mlp.h>
#include <rnn.h>
#include <lstm.h>
#include <rs.h>
#include <hopper_env.h>
#include <string.h>
#include <locale.h>

#define NETWORK_TYPE MLP
#define ROLLOUTS_PER_MEMBER 1
#define MAX_TRAJ_LEN 300

size_t samples = 0;

float evaluate(Environment *env, NETWORK_TYPE *n, int render){
	float perf = 0;
	for(int i = 0; i < ROLLOUTS_PER_MEMBER; i++){
		env->reset(*env);
		env->seed(*env);

		for(int t = 0; t < MAX_TRAJ_LEN; t++){
			samples++;
			mlp_forward(n, env->state);
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

int main(int argc, char **argv){
  setlocale(LC_ALL,"");

	Environment env = create_hopper_env();
	MLP seed = create_mlp(env.observation_space, 10, env.action_space);
	for(int i = 0; i < seed.depth; i++)
		seed.layers[i].logistic = linear;

	for(int j = 0; j < seed.num_params; j++)
		seed.params[j] = 0;

	srand(time(NULL));
	RS r = create_rs(seed.params, seed.num_params, 60);
	r.cutoff = 0.66;
	r.step_size = 0.01;
	r.std = 0.0025;
	r.algo = V1;
	
  size_t episodes = 0;
	int iter = 0;
	while(samples < 1e7){
		iter++;
		for(int i = 0; i < r.directions; i++){
			for(int j = 0; j < seed.num_params; j++)
				seed.params[j] += 1 * r.deltas[i]->p[j];
			r.deltas[i]->r_pos = evaluate(&env, &seed, 0);

			for(int j = 0; j < seed.num_params; j++)
				seed.params[j] -= 2 * r.deltas[i]->p[j];
			r.deltas[i]->r_neg = evaluate(&env, &seed, 0);
		
			for(int j = 0; j < seed.num_params; j++)
				seed.params[j] += 1 * r.deltas[i]->p[j];

      episodes += 2 * ROLLOUTS_PER_MEMBER;
		}
		printf("iteration %3d | reward: %8.4f | episodes %7lu | samples %'9lu \n", iter, evaluate(&env, &seed, !(iter%50)), episodes, samples);
		rs_step(r);

		save_mlp(&seed, "./model/search.mlp");
	}
	return 0;
}
