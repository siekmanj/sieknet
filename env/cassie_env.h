#ifndef CASSIE_ENV_H
#define CASSIE_ENV_H

#include <cassiemujoco.h>
#include <mujoco.h>
#include <glfw3.h>
#include <env.h>

#define CASSIE_ENV_REFTRAJ

typedef struct data {
  cassie_sim_t *sim;
	cassie_vis_t *vis;
  state_out_t est;

  mjvCamera camera;
  mjvOption opt;
  mjvScene scene;
  mjrContext context;
  GLFWwindow *window;

#ifdef CASSIE_ENV_REFTRAJ
	/* 
	 * traj[t][0]      time   (state)
	 * traj[t][ 1..36] qpos   (state)
	 * traj[t][36..68] qvel   (state)
	 * traj[t][68..78] torque (action)
	 * traj[t][78..88] mpos   (action)
	 * traj[t][88..98] mvel   (action)
	 */

	double **traj;
#endif

	size_t counter;
	size_t phaselen;
	size_t phase;
	size_t time;
  double real_dt;
	int render_setup;
} Data;

Environment create_cassie_env();

#endif
