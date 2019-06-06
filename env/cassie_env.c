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

//#define CASSIE_ENV_USE_REF_TRAJ

#define CASSIE_ENV_NO_DELTAS

//#define CASSIE_ENV_USE_HUMANOID_REWARD

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

void cassie_env_dispose(Environment env){}

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


#elif defined(CASSIE_ENV_USE_REF_TRAJ)
	//env.observation_space = 80;
  #error "Not implemented"

#endif

}

void cassie_env_reset(Environment env){
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

void cassie_env_seed(Environment env){
	Data *tmp = (Data*)env.data;
  mjData *d = cassie_sim_mjdata(tmp->sim);

	for(int i = 0; i < REF_QPOS_LEN; i++)
		d->qpos[i] += uniform(-0.005, 0.005);

	for(int i = 0; i < REF_QVEL_LEN; i++)
		d->qvel[i] += uniform(-0.005, 0.005);

	set_state(env);
}


#define USE_CASSIE_VIS
void cassie_env_render(Environment env){
	Data *tmp  = (Data*)env.data;

#ifndef USE_CASSIE_VIS
  mjData *d  = cassie_sim_mjdata(tmp->sim);
  mjModel *m = cassie_sim_mjmodel(tmp->sim);

	if(!tmp->render_setup){
    tmp->window = glfwCreateWindow(1200, 900, "MuJoCo", NULL, NULL);
    glfwMakeContextCurrent(tmp->window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&tmp->camera);
    mjv_defaultOption(&tmp->opt);
    mjv_defaultScene(&tmp->scene);
    mjr_defaultContext(&tmp->context);

    mjv_makeScene(m, &tmp->scene, 2000);
    mjr_makeContext(m, &tmp->context, mjFONTSCALE_150);
    tmp->render_setup = 1;
	}

  for(int i = 0; i < 0; i++)
    tmp->camera.lookat[i] = d->qpos[i];

  tmp->camera.distance = 3;
  tmp->camera.elevation = -20.0;

  mjrRect viewport = {0, 0, 0, 0};
  glfwMakeContextCurrent(tmp->window);
  glfwGetFramebufferSize(tmp->window, &viewport.width, &viewport.height);

  mjv_updateScene(m, d, &tmp->opt, NULL, &tmp->camera, mjCAT_ALL, &tmp->scene);
  mjr_render(viewport, &tmp->scene, &tmp->context);

  // swap OpenGL buffers (blocking call due to v-sync)
  glfwSwapBuffers(tmp->window);

  // process pending GUI events, call GLFW callbacks
  glfwPollEvents();
#else
	if(!tmp->render_setup){
    tmp->vis = cassie_vis_init(tmp->sim, "assets/cassie.xml");
    tmp->render_setup = 1;
  }
	cassie_vis_draw(tmp->vis, tmp->sim);
#endif

  /* Ensure realtime rendering */
  if(tmp->real_dt < dt(env)){
    struct timespec sleep_for = {0, (long)(1e9 * (dt(env) - tmp->real_dt))};
    nanosleep(&sleep_for, NULL);
  }

}

static void cassie_env_close(Environment env){
	Data *tmp = (Data*)env.data;
#ifndef USE_CASSIE_VIS
  if(tmp->render_setup){
    glfwDestroyWindow(tmp->window);
    tmp->render_setup = 0;
  }
#else
	cassie_vis_close(tmp->vis);
  tmp->render_setup = 0;
#endif
}

static float calculate_reward(Environment env){
	Data *tmp  = (Data*)env.data;
  mjData *d  = cassie_sim_mjdata(tmp->sim);

#ifndef CASSIE_ENV_USE_HUMANOID_REWARD
  // Use a reward based on matching the expert trajectory
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

  double last_qpos[REF_QPOS_LEN];
  get_ref_qpos_raw(tmp->traj, env.frameskip, 28, last_qpos);

	double expected_x = last_qpos[0] * tmp->counter + ref_qpos[0];
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
  // Use the OpenAI-gym humanoid-v1 reward 
  float lin_vel_cost = 1.25 * (d->qvel[0]) / dt(env);

  float quad_ctrl_cost = 0;
  for(int i = 0; i < env.action_space; i++)
    quad_ctrl_cost += 0.1 * d->ctrl[i] * d->ctrl[i];

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

#ifdef CASSIE_ENV_NO_DELTAS
  double offsets[] = {0.0045, 0.0000, 0.4973, -1.9997, -1.5968, 0.0045, 0.0000, 0.4973, -1.997, -1.5968};
#endif

	pd_in_t u = {0};
	for(int i = 0; i < 5; i++){
#ifndef CASSIE_ENV_NO_DELTAS
		double ltarget = action[i+0] + ref_pos[i+0];
		double rtarget = action[i+5] + ref_pos[i+5];
#else
		double ltarget = action[i+0] + offsets[i+0];
		double rtarget = action[i+5] + offsets[i+5];
#endif

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

float cassie_env_step(Environment env, float *action){
	Data *tmp = (Data*)env.data;

  clock_t start = clock();

  mjData *d  = cassie_sim_mjdata(tmp->sim);

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

#ifndef CASSIE_ENV_USE_HUMANOID_REWARD
	if(reward < 0.3)
		*env.done = 1;
#endif

	set_state(env);

  tmp->real_dt = (double)(clock() - start)/CLOCKS_PER_SEC;
  return reward;
}

Environment create_cassie_env(){

#ifndef USE_CASSIE_VIS
  glfwInit();
#endif
	setenv("MUJOCO_KEY_PATH", SIEKNET_MJKEYPATH, 0);
	setenv("CASSIE_MODEL_PATH", "assets/cassie.xml", 1);

  const char modelfile[] = "assets/cassie.xml";
	const char trajfile[]  = "assets/stepdata.bin";

  if(!cassie_mujoco_init(modelfile)){
    printf("WARNING: create_cassie_env(): cassie_mujoco_init() returned 0.\n");
  }

  cassie_sim_t *c = cassie_sim_init(modelfile);

  Environment env;
  
  env.render = cassie_env_render;
  env.close = cassie_env_close;
  env.step = cassie_env_step;
  env.dispose = cassie_env_dispose;
  env.reset = cassie_env_reset;
  env.seed = cassie_env_seed;

	env.frameskip = 60;
  env.alive_bonus = 0.0f;

  Data *d = (Data*)malloc(sizeof(Data));

	d->sim = c;
	d->vis = NULL;
  d->render_setup = 1;
	d->render_setup = 0;
	d->counter = 0;
	d->phase = 0;
	d->time = 0;

#ifndef CASSIE_ENV_USE_HUMANOID_REWARD
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
		size_t num_read = fread(d->traj[i], sizeof(double), traj_data_row_len, fp);

    /*
    if(n_read != traj_data_row_len){
      printf("WARNING: create_cassie_env(): may not have been able to read stepdata.bin correctly, ", n_read, traj_data_row_len);
      if(ferror(fp))
        perror("error encountered");
      else if(feof(fp))
        perror("got an EOF");
    }
    */
  }
	d->phaselen = (size_t)(TRAJECTORY_LENGTH / env.frameskip);
#else
  d->traj = NULL;
#endif

	env.data = d;

	env.observation_space = 40;

#if defined(CASSIE_ENV_USE_CLOCK)

  env.observation_space += 2;

#elif defined(CASSIE_ENV_USE_REF_TRAJ)

  env.observation_space += 40;

  #error "Not implemented"

#endif

  env.action_space = 10;
	env.state = calloc(env.observation_space, sizeof(float));

  env.done = calloc(1, sizeof(int));
  return env;
}

