#include <mj_env.h>
#include <stdio.h>

static float step(Environment env, float *action){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;

  float posbefore = d->qpos[0];

  mjtNum simstart = d->time;
  for(int i = 0; i < env.action_space; i++)
    d->ctrl[i] = action[i];
  
  for(int i = 0; i < env.frameskip; i++)
    mj_step(m, d);

  for(int i = tmp->qpos_start; i < m->nq; i++)
    env.state[i - tmp->qpos_start] = d->qpos[i];

  for(int i = 0; i < m->nv; i++)
    env.state[i + m->nq - tmp->qpos_start] = d->qvel[i];

  for(int i = 0; i < env.observation_space; i++){
    if(isnan(env.state[i])){
      printf("\nWARNING: NaN in observation vector - aborting episode early.\n");
      *env.done = 1;
      return 0;
    }
  }

  /* REWARD CALCULATION: Identical to OpenAI's */
  
  float lin_vel_cost = 1.25 * (d->qpos[0] - posbefore) / (d->time - simstart);

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

  if(d->qpos[2] < 1.0 || d->qpos[2] > 2.0){
    *env.done = 1;
  }

  if(!isfinite(reward)){
    *env.done = 1;
    reward = 0;
  }

  return reward;
}

Environment create_humanoid_env(){
  Environment ret = create_mujoco_env("./assets/humanoid.xml", step, 2);
  ret.alive_bonus = 5.0f;
  ret.frameskip = 5;
  return ret;
}


