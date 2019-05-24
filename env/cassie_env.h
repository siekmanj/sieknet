#ifndef CASSIE_ENV_H
#define CASSIE_ENV_H

#include <cassiemujoco.h>
#include <env.h>

typedef struct data {
  cassie_sim_t *sim;
	cassie_vis_t *vis;
	float **traj;

	float *p_gains;
	float *d_gains;

	size_t phase;
	size_t t;
	int render_setup;
} Data;

Environment create_cassie_env();

#endif
