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

  for(int i = 1; i < m->nq; i++)
    env.state[i-1] = d->qpos[i];

  for(int i = 0; i < m->nv; i++)
    env.state[i + m->nq - 1] = d->qvel[i];

  /* REWARD CALCULATION: Identical to OpenAI's */
  
  float reward = (d->qpos[0] - posbefore) / (d->time - simstart);
  reward += ALIVE_BONUS;

  float action_sum = 0;
  for(int i = 0; i < env.action_space; i++)
    action_sum += action[i]*action[i];

  reward -= 0.001 * action_sum;

  if(d->qpos[1] < 0.7 || d->qpos[2] < -1 || d->qpos[2] > 1){
    *env.done = 1;
  }

  return reward;
}

Environment create_hopper_env(){
  return create_mujoco_env("./assets/hopper.xml", step, 1);
}
