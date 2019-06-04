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

  /* REWARD CALCULATION: Identical to OpenAI's */
  
  float forward_reward = (d->qpos[0] - posbefore) / (d->time - simstart);

  float ctrl_cost = 0;
  for(int i = 0; i < env.action_space; i++)
    ctrl_cost += 0.5 * action[i] * action[i];

  float contact_cost = 0;
  for(int i = 0; i < m->nbody; i++){
    float contact_force = d->cfrc_ext[i];
    if(contact_force >  1) contact_force =  1;
    if(contact_force < -1) contact_force = -1;

    contact_cost += 0.5 * 1e-3 * contact_force * contact_force;
  }

  float survive_reward = env.alive_bonus;

  float reward = forward_reward - ctrl_cost - contact_cost + survive_reward;

  for(int i = 0; i < env.observation_space; i++){
    if(isnan(env.state[i])){
      printf("\nWARNING: NaN in observation vector - aborting episode early.\n");
      *env.done = 1;
      return 0;
    }
  }
  if(d->qpos[2] < 0.2 || d->qpos[2] > 1.0){
    *env.done = 1;
  }

  return reward;
}

Environment create_ant_env(){
  Environment ret = create_mujoco_env("./assets/ant.xml", step, 2);
  ret.alive_bonus = 1.0f;
  ret.frameskip = 5;
  return ret;
}

