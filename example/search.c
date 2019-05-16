#include <conf.h>
#include <optimizer.h>
#include <mlp.h>
#include <rnn.h>
#include <lstm.h>
#include <rs.h>
#include <hopper_env.h>
#include <string.h>

#define NETWORK_TYPE MLP
#define ROLLOUTS_PER_MEMBER 2
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

				//sensitivity(n);
				//abs_backward(network_type)(n);
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
	Environment env = create_hopper_env();
	MLP seed = create_mlp(env.observation_space, 16, env.action_space);
	for(int i = 0; i < seed.depth; i++)
		seed.layers[i].logistic = linear;

	for(int j = 0; j < seed.num_params; j++)
		seed.params[j] = 0;

	RS r = create_rs(seed.params, seed.num_params, 230);
	r.std = 0.0075;
	
	float *update = ALLOC(float, seed.num_params);

	SGD o = cpu_init_SGD(seed.params, update, seed.num_params);
	o.learning_rate = 0.1;

	while(samples < 1e8){
		memset(update, '\0', sizeof(float)*seed.num_params);

		for(int i = 0; i < r.directions; i++){
			for(int j = 0; j < seed.num_params; j++)
				seed.params[j] += 1*r.perturbances[i][j];
			r.r_pos[i] = evaluate(&env, &seed, 0);

			for(int j = 0; j < seed.num_params; j++)
				seed.params[j] -= 2*r.perturbances[i][j];
			r.r_neg[i] = evaluate(&env, &seed, 0);
		
			for(int j = 0; j < seed.num_params; j++)
				seed.params[j] += 1*r.perturbances[i][j];
			
		}
		for(int i = 0; i < r.directions; i++){
			float weight = -(r.step_size / r.directions) * (r.r_pos[i] - r.r_neg[i]);
			for(int j = 0; j < seed.num_params; j++)
				update[j] += weight * r.perturbances[i][j];
			
		}
		//PRINTLIST(update, seed.num_params);
		printf("reward: %f, samples %lu\n", evaluate(&env, &seed,1), samples);
		o.step(o);
		PRINTLIST(seed.params, seed.num_params);
		rs_step(r);
	}
	return 0;
}
