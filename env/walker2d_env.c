#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mj_env.h>
#define ALIVE_BONUS 0.0f
#define FRAMESKIP 5

static float step(Environment env, float *action){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;

  float posbefore = d->qpos[0];

  mjtNum simstart = d->time;
  for(int i = 0; i < env.action_space; i++)
    d->ctrl[i] = action[i];
  
  for(int i = 0; i < FRAMESKIP; i++)
    mj_step(m, d);

  for(int i = tmp->qpos_start; i < m->nq; i++)
    env.state[i - tmp->qpos_start] = d->qpos[i];

  for(int i = 0; i < m->nv; i++)
    env.state[i + m->nq - tmp->qpos_start] = d->qvel[i];

  /* REWARD CALCULATION: Similar to OpenAI's */
  
  float reward = (d->qpos[0] - posbefore) / (d->time - simstart);
  reward += ALIVE_BONUS / FRAMESKIP;

  float action_sum = 0;
  for(int i = 0; i < env.action_space; i++)
    action_sum += action[i]*action[i];

  reward -= 0.005 * action_sum;

  if(d->qpos[1] < 0.8 || d->qpos[1] > 2.0 || d->qpos[2] < -1.0 || d->qpos[2] > 1.0){
    *env.done = 1;
  }

  return reward;
}

Environment create_walker2d_env(){
  return create_mujoco_env("./assets/walker2d.xml", step, 1);
}
