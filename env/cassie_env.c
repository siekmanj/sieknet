#include <cassiemujoco.h>
#include <cassie_env.h>
#include <mujoco.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <conf.h>

#define CASSIE_ENV_USE_CLOCK
#define CASSIE_ENV_USE_REF_TRAJ

static const double JOINT_WEIGHTS[] = {0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05};

static const size_t ACTION_POS_IDX[10] = {7, 8, 9, 14, 20, 21, 22, 23, 28, 34};
static const size_t ACTION_VEL_IDX[10] = {6, 7, 8, 12, 18, 19, 20, 21, 25, 31};

static const size_t STATE_POS_IDX[20] = {1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34};
static const size_t STATE_VEL_IDX[20] = {0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31};

static const float PID_P[5] = {100,  100,  88,  96,  50};
static const float PID_D[5] = {10.0, 10.0, 8.0, 9.6, 5.0};

const size_t TRAJECTORY_LENGTH = 1684; /* 1684 rows in stepdata.bin */

#define LENGTHOF(arr) (sizeof(arr)/sizeof(arr[0]))

#define REF_QPOS_START  1
#define REF_QPOS_END   36
const size_t REF_QPOS_LEN   = REF_QPOS_END - REF_QPOS_START;

#define REF_QVEL_START 36
#define REF_QVEL_END   68
const size_t REF_QVEL_LEN   = REF_QVEL_END - REF_QVEL_START;

#define REF_TORQUE_START 68
#define REF_TORQUE_END   78
const size_t REF_TORQUE_LEN = REF_TORQUE_END - REF_TORQUE_START;

#define REF_MPOS_START 78
#define REF_MPOS_END   88
const size_t REF_MPOS_LEN   = REF_MPOS_END - REF_MPOS_START;

#define REF_MVEL_START 88
#define REF_MVEL_END   98
const size_t REF_MVEL_LEN   = REF_MVEL_END - REF_MVEL_START;

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float dt(Environment env){
	return (float) 1 / 2000 * env.frameskip;
}

static void get_ref_qpos_raw(double **traj, size_t frameskip, size_t phase, double *dest){

  double *raw_qpos = &traj[phase * frameskip][REF_QPOS_START];
  for(int i = 0; i < REF_QPOS_LEN; i++){
    dest[i] = raw_qpos[i];
  }
}

static void get_ref_qpos_state(double **traj, size_t frameskip, size_t phase, double *dest){

	double tmp[REF_QPOS_LEN];
	get_ref_qpos_raw(traj, frameskip, phase, tmp);

	for(int i = 0; i < LENGTHOF(STATE_POS_IDX); i++){
		if(!i)
			dest[i] = 0.0f;
		else
			dest[i] = tmp[STATE_POS_IDX[i]];
	}
}

static void get_ref_qpos_action(double **traj, size_t frameskip, size_t phase, double *dest){
	double tmp[REF_QPOS_LEN];
	get_ref_qpos_raw(traj, frameskip, phase, tmp);

	for(int i = 0; i < LENGTHOF(ACTION_POS_IDX); i++){
		dest[i] = tmp[ACTION_POS_IDX[i]];
	}
}

static void get_ref_qvel_raw(double **traj, size_t frameskip, size_t phase, double *dest){

	double *raw_qvel = &traj[phase * frameskip][REF_QVEL_START];
	for(int i = 0; i < REF_QVEL_LEN; i++){
		dest[i] = raw_qvel[i];
	}
}

static void get_ref_qvel_state(double **traj, size_t frameskip, size_t phase, double *dest){
	double tmp[REF_QVEL_LEN];
	get_ref_qvel_raw(traj, frameskip, phase, tmp);

	for(int i = 0; i < LENGTHOF(STATE_VEL_IDX); i++){
		dest[i] = tmp[STATE_VEL_IDX[i]];
	}
}

void dispose(Environment env){

}

static void set_state(Environment env){
	Data *tmp = (Data*)env.data;
  mjData *d = cassie_sim_mjdata(tmp->sim);

	for(int i = 0; i < 20; i++){
		env.state[i] = d->qpos[STATE_POS_IDX[i]];
	}

	for(int i = 0; i < 20; i++){
			env.state[i+20] = d->qvel[STATE_VEL_IDX[i]];
	}

#if defined(CASSIE_ENV_USE_CLOCK)
	double sin_clock = sin(2 * M_PI * (double)tmp->phase / tmp->phaselen);
	double cos_clock = cos(2 * M_PI * (double)tmp->phase / tmp->phaselen);
	env.state[env.observation_space - 2] = sin_clock;
	env.state[env.observation_space - 1] = cos_clock;


#elif defined(CASSIE_ENV_REF)
	env.observation_space = 80;
  #error "Not implemented"

#elif defined(CASSIE_ENV_NOCLOCK)

  //nothing

#endif

}

void reset(Environment env){
	*env.done = 0;
	Data *tmp = (Data*)env.data;

	tmp->counter = 0;
	tmp->phase = 0;//rand() % tmp->phaselen;
	tmp->time  = 0;

  mjData *d = cassie_sim_mjdata(tmp->sim);
	d->time = 0;

	double *qpos = d->qpos;
	double *qvel = d->qvel;

	get_ref_qpos_raw(tmp->traj, env.frameskip, tmp->phase, qpos);
	get_ref_qvel_raw(tmp->traj, env.frameskip, tmp->phase, qvel);

	set_state(env);
}

void seed(Environment env){
	Data *tmp = (Data*)env.data;
  mjData *d = cassie_sim_mjdata(tmp->sim);

	for(int i = 0; i < REF_QPOS_LEN; i++)
		d->qpos[i] += uniform(-0.005, 0.005);

	for(int i = 0; i < REF_QVEL_LEN; i++)
		d->qvel[i] += uniform(-0.005, 0.005);

	set_state(env);
}

void render(Environment env){
	Data *tmp = (Data*)env.data;
	if(!tmp->vis){
		tmp->render_setup = 1;
    tmp->vis = cassie_vis_init(tmp->sim, "assets/cassie.xml");
	}
	cassie_vis_draw(tmp->vis, tmp->sim);
}

static void env_close(Environment env){
	Data *tmp = (Data*)env.data;
	cassie_vis_close(tmp->vis);
  tmp->render_setup = 0;
}

static float calculate_reward(Environment env){
	Data *tmp = (Data*)env.data;
  mjData *d = cassie_sim_mjdata(tmp->sim);

#ifdef CASSIE_ENV_USE_REF_TRAJ
  /* Use a reward based on matching the expert trajectory */
	double ref_qpos[REF_QPOS_LEN];
	get_ref_qpos_raw(tmp->traj, env.frameskip, tmp->phase, ref_qpos);

	double joint_error       = 0;
	double com_error         = 0;
	double orientation_error = 0;
	double spring_error      = 0;

	for(int i = 0; i < LENGTHOF(ACTION_POS_IDX); i++){
		double target = ref_qpos[ACTION_POS_IDX[i]];
		double actual = d->qpos[ACTION_POS_IDX[i]];

		joint_error += 30 * JOINT_WEIGHTS[i] * (target - actual) * (target - actual);
	}

	double expected_x = (tmp->traj[TRAJECTORY_LENGTH-2][1] - tmp->traj[0][1]) * tmp->counter;
	double expected_y = 0;
	double expected_z = ref_qpos[2];

	double actual_x = d->qpos[0];
	double actual_y = d->qpos[1];
	double actual_z = d->qpos[2];

	com_error += (expected_x - actual_x) * (expected_x - actual_x);
	com_error += (expected_y - actual_y) * (expected_y - actual_y);
	com_error += (expected_z - actual_z) * (expected_z - actual_z);

	for(int i = 4; i < 7; i++){
		double target = ref_qpos[i];
		double actual = d->qpos[i];

		orientation_error += (target - actual) * (target - actual);
	}

	for(int i = 15; i < 29; i++){
		double target = ref_qpos[i];
		double actual = d->qpos[i];

		spring_error += 1000 * (target - actual) * (target - actual);
	}
	
	joint_error       = 0.5 * exp(-joint_error);
	com_error         = 0.3 * exp(-com_error);
	orientation_error = 0.1 * exp(-orientation_error);
	spring_error      = 0.1 * exp(-spring_error);

	double reward = joint_error + com_error + orientation_error;
#else
  /* Use the OpenAI-gym humanoid-v1 reward */
  float lin_vel_cost = 1.25 * (d->qvel[0]) / (d->time - simstart);

  float quad_ctrl_cost = 0;
  for(int i = 0; i < env.action_space; i++)
    quad_ctrl_cost += 0.1 * action[i] * action[i];

  float quad_impact_cost = 0;
  for(int i = 0; i < m->nbody; i++){
    float contact_force = d->cfrc_ext[i];
    quad_impact_cost += 5e-7 * contact_force * contact_force;
  }
  if(quad_impact_cost > 10)
    quad_impact_cost = 10;

  float reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + env.alive_bonus;
#endif

	return reward;
}

static void sim_step(Environment env, float *action){
	Data *tmp = (Data*)env.data;

	size_t next_phase = tmp->phase + 1;
	if(next_phase > tmp->phaselen)
		next_phase = 0;

	double ref_pos[LENGTHOF(ACTION_POS_IDX)];
	get_ref_qpos_action(tmp->traj, env.frameskip, next_phase, ref_pos);

	pd_in_t u;
	for(int i = 0; i < 5; i++){
		double ltarget = action[i+0] + ref_pos[i+0];
		double rtarget = action[i+5] + ref_pos[i+5];

    //printf("%d: ltarget: %f + %f, rtarget: %f + %f\n", i, action[i], ref_pos[i], action[i+5], ref_pos[i+5]);

		u.leftLeg.motorPd.pGain[i]  = PID_P[i];
		u.rightLeg.motorPd.pGain[i] = PID_P[i];

		u.leftLeg.motorPd.dGain[i]  = PID_D[i];
		u.rightLeg.motorPd.dGain[i] = PID_D[i];

		u.leftLeg.motorPd.torque[i]  = 0;
		u.rightLeg.motorPd.torque[i] = 0;

		u.leftLeg.motorPd.pTarget[i]  = ltarget;
		u.rightLeg.motorPd.pTarget[i] = rtarget;

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

	/*
	double ref_pos[LENGTHOF(ACTION_POS_IDX)];
	get_ref_qpos_action(tmp->traj, env.frameskip, tmp->phase+1, ref_pos);
	printf("phase %d: ", tmp->phase);
	PRINTLIST(ref_pos, LENGTHOF(ACTION_POS_IDX));
	getchar();
	*/

	tmp->time++;
	tmp->phase++;

	if(tmp->phase > tmp->phaselen){
		tmp->phase = 0;
		tmp->counter++;
	}
	
	if(d->qpos[2] <= 0.5 || d->qpos[2] > 3.0){
		*env.done = 1;
	}
	double reward = calculate_reward(env);

	if(reward < 0.3)
		*env.done = 1;

	set_state(env);
  return reward;
}

Environment create_cassie_env(){
	setenv("MUJOCO_KEY_PATH", SIEKNET_MJKEYPATH, 0);
	setenv("CASSIE_MODEL_PATH", "assets/cassie.xml", 0);

  const char modelfile[] = "assets/cassie.xml";
	const char trajfile[]  = "assets/stepdata.bin";

  cassie_sim_t *c = cassie_sim_init(modelfile);

  Environment env;
  
  env.render = render;
  env.close = env_close;
  env.step = step;
  env.dispose = dispose;
  env.reset = reset;
  env.seed = seed;

	env.frameskip = 60;
  env.alive_bonus = 0.0f;

  Data *d = (Data*)malloc(sizeof(Data));

	d->sim = c;
	d->vis = NULL;
	d->render_setup = 0;
	d->counter = 0;
	d->phase = 0;
	d->time = 0;

	size_t idx = 0;

	FILE *fp = fopen(trajfile, "rb");
	if(!fp){
		printf("ERROR: create_cassie_env(): couldn't open binary file '%s'\n", trajfile);
		exit(1);
	}
	fseek(fp, 0L, SEEK_SET);

	size_t traj_data_row_len = 1 + 35 + 32 + 10 + 10 + 10;

	d->traj = ALLOC(double*, TRAJECTORY_LENGTH); 
  for(int i = 0; i < TRAJECTORY_LENGTH; i++){

		d->traj[i] = ALLOC(double, traj_data_row_len);
		size_t n_read = fread(d->traj[i], sizeof(double), traj_data_row_len, fp);

    if(n_read != traj_data_row_len){
      printf("WARNING: create_cassie_env(): unable to read stepdata.bin correctly, read %lu of %lu bytes\n", n_read, traj_data_row_len);
      //exit(1);
    }
  }
	d->phaselen = (size_t)(TRAJECTORY_LENGTH / env.frameskip);

	env.data = d;

#if defined(CASSIE_ENV_USE_CLOCK)

  env.observation_space = 42;

#elif defined(CASSIE_ENV_REF)

  env.observation_space = 80;

  #error "Not implemented"

#elif defined(CASSIE_ENV_NOCLOCK)

  env.observation_space = 40;

#else

  #error "Cassie environment type not defined"

#endif

  env.action_space = 10;
	env.state = calloc(env.observation_space, sizeof(float));

  env.done = calloc(1, sizeof(int));
  return env;
}

