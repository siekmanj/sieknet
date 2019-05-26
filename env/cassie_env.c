#include <cassiemujoco.h>
#include <cassie_env.h>
#include <mujoco.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float dt(Environment env){
	return (float) 1 / 2000 * env.frameskip;
}

void dispose(Environment env){

}

void reset(Environment env){
	//set_qpos
	//set_qvel
	//get_full_state
}

void seed(Environment env){

}

void render(Environment env){
	Data *tmp = (Data*)env.data;
	if(!tmp->vis){
		tmp->render_setup = 1;
    tmp->vis = cassie_vis_init(tmp->sim, "assets/cassie.xml");
	}
	cassie_vis_draw(tmp->vis, tmp->sim);
}

static void close(Environment env){
	Data *tmp = (Data*)env.data;
	cassie_vis_close(tmp->vis);
  tmp->render_setup = 0;
}

static float sim_step(Environment env, float *action){
	Data *tmp = (Data*)env.data;

	pd_in_t u;
	for(int i = 0; i < 5; i++){
		u.leftLeg.motorPd.pGain[i]  = tmp->p_gains[i];
		u.rightLeg.motorPd.pGain[i] = tmp->p_gains[i];

		u.leftLeg.motorPd.dGain[i]  = tmp->d_gains[i];
		u.rightLeg.motorPd.dGain[i] = tmp->d_gains[i];

		u.leftLeg.motorPd.torque[i]  = 0;
		u.rightLeg.motorPd.torque[i] = 0;

		u.leftLeg.motorPd.pTarget[i]  = action[i];
		u.rightLeg.motorPd.pTarget[i] = action[i+5];

		u.leftLeg.motorPd.dTarget[i]  = 0;
		u.rightLeg.motorPd.dTarget[i] = 0;
	}
	state_out_t y;
	cassie_sim_step_pd(tmp->sim, &y, &u);
}

float step(Environment env, float *action){
	Data *tmp = (Data*)env.data;
  cassie_sim_t *c = tmp->sim;

  mjData *d = cassie_sim_mjdata(c);

	for(int i = 0; i < env.frameskip; i++){
		sim_step(env, action);
	}

	//calc reward
  return 0.0f;
}
Environment create_cassie_env(){
	setenv("MUJOCO_KEY_PATH", "/home/jonah/.mujoco/mjkey.txt", 0);
	setenv("CASSIE_MODEL_PATH", "assets/cassie.xml", 0);

  const char modelfile[] = "assets/cassie.xml";
  cassie_sim_t *c = cassie_sim_init(modelfile);
  //cassie_vis_t *v = cassie_vis_init(c, modelfile);

  Environment env;
  
	float pid_p[5] = {100,  100,  88,  96,  50};
	float pid_d[5] = {10.0, 10.0, 8.0, 9.6, 5.0};

	size_t pos_idx[10] = {7, 8, 9, 14, 20, 21, 22, 23, 28, 34};
	size_t vel_idx[10] = {6, 7, 8, 12, 18, 19, 20, 21, 25, 31};

  env.render = render;
  env.close = close;
  env.step = step;
  env.dispose = dispose;
  env.reset = reset;
  env.seed = seed;

	env.frameskip = 60;

  Data *d = (Data*)malloc(sizeof(Data));

	d->sim = c;
	d->vis = NULL;
	d->render_setup = 0;

	d->p_gains = (float*)malloc(5 * sizeof(float));
	d->d_gains = (float*)malloc(5 * sizeof(float));

	memcpy(d->p_gains, pid_p, 5 * sizeof(float));
	memcpy(d->d_gains, pid_d, 5 * sizeof(float));

	env.data = d;

  env.state = NULL;

  env.observation_space = 0; //TODO
  env.action_space = 10; //TODO

  env.done = calloc(1, sizeof(int));
	printf("made it out!\n");

  return env;
}

