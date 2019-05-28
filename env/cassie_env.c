#include <cassiemujoco.h>
#include <cassie_env.h>
#include <mujoco.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <conf.h>

#define CASSIE_ENV_USE_CLOCK
#define CASS_ENV_USE_REF_TRAJ

static const double JOINT_WEIGHTS[] = {0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05};

static const size_t POS_IDX[10] = {7, 8, 9, 14, 20, 21, 22, 23, 28, 34};
static const size_t VEL_IDX[10] = {6, 7, 8, 12, 18, 19, 20, 21, 25, 31};

static const size_t STATE_POS_IDX[20] = {1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34};
static const size_t STATE_VEL_IDX[20] = {0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31};

static const float PID_P[5] = {100,  100,  88,  96,  50};
static const float PID_D[5] = {10.0, 10.0, 8.0, 9.6, 5.0};

const size_t TRAJECTORY_LENGTH = 1684; /* 1684 rows in stepdata.bin */

#define REF_QPOS_START  1
#define REF_QPOS_END   36
const size_t REF_QPOS_LEN   = REF_QPOS_END - REF_QPOS_START;

#define REF_QVEL_START 36
#define REF_QVEL_END   68
const size_t REF_QVEL_LEN   = REF_QVEL_END - REF_QVEL_START;

#define REF_TORQUE_START 68
#define REF_TORQUE_END   78
const size_t REF_TORQUE_LEN   = REF_TORQUE_END - REF_TORQUE_START;

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

static double *get_ref_pos(Environment env){
	Data *tmp = (Data*)env.data;
	
	return &tmp->traj[tmp->phase * env.frameskip][REF_QPOS_START];

}

static double *get_ref_vel(Environment env){
	Data *tmp = (Data*)env.data;

	return &tmp->traj[tmp->phase * env.frameskip][REF_QVEL_START];
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
	tmp->phase = rand() % tmp->phaselen;
	tmp->time  = 0;

  mjData *d = cassie_sim_mjdata(tmp->sim);

	double *qpos = d->qpos;
	double *qvel = d->qvel;

	double *ref_qpos = get_ref_pos(env);
	double *ref_qvel = get_ref_vel(env);

	for(int i = 0; i < REF_QPOS_LEN; i++){
		if(!i)
			qpos[i] = 0;
		else
			qpos[i] = ref_qpos[i];
	}
	for(int i = 0; i < REF_QVEL_LEN; i++){
		qvel[i] = ref_qvel[i];
	}
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

static void close(Environment env){
	Data *tmp = (Data*)env.data;
	cassie_vis_close(tmp->vis);
  tmp->render_setup = 0;
}

static float calculate_reward(Environment env){
	Data *tmp = (Data*)env.data;
  mjData *d = cassie_sim_mjdata(tmp->sim);

	double *ref_qpos = get_ref_pos(env);
	//double *ref_qvel = get_ref_vel(env);

	double joint_error       = 0;
	double com_error         = 0;
	double orientation_error = 0;
	//double spring_error      = 0;

	for(int i = 0; i < sizeof(POS_IDX)/sizeof(POS_IDX[0]); i++){
		double target = ref_qpos[POS_IDX[i]];
		double actual = d->qpos[i];

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
	
	joint_error       = 0.5 * exp(-joint_error);
	com_error         = 0.3 * exp(-com_error);
	orientation_error = 0.1 * exp(-orientation_error);

	double reward = joint_error + com_error + orientation_error;

	return reward;
}

static void sim_step(Environment env, float *action){
	Data *tmp = (Data*)env.data;

	double *ref_pos = tmp->traj[tmp->phase + 1];
	//double *ref_vel = tmp->traj[tmp->phase + 1];

	pd_in_t u;
	for(int i = 0; i < 5; i++){
		float ltarget = action[i+0] + ref_pos[POS_IDX[i+0]];
		float rtarget = action[i+5] + ref_pos[POS_IDX[i+5]];

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
	d->counter = 0;
	d->phase = 0;
	d->time = 0;

	size_t idx = 0;

	FILE *fp = fopen(trajfile, "rb");
	if(!fp){
		printf("ERROR: create_cassie_env(): couldn't open binary file '%s'\n", trajfile);
		exit(1);
	}

	d->traj = ALLOC(double*, TRAJECTORY_LENGTH); 
	int traj_data_row_len = 1 + 35 + 32 + 10 + 10 + 10;
	size_t n_read = 0;

	do{

		d->traj[idx] = ALLOC(double, traj_data_row_len);
		n_read = fread(d->traj[idx], sizeof(double), traj_data_row_len, fp);
		idx++;

	}while(n_read > 0);

	d->phaselen = (size_t)(TRAJECTORY_LENGTH / env.frameskip);

	free(d->traj[idx]);

	env.data = d;

  env.state = NULL;

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

