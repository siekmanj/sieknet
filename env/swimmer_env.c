#include <mj_env.h>

#define FRAMESKIP 3

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

  /* REWARD CALCULATION: Identical to OpenAI's */
  
  float reward = (d->qpos[0] - posbefore) / (d->time - simstart);

  float action_sum = 0;
  for(int i = 0; i < env.action_space; i++)
    action_sum += action[i]*action[i];

  reward -= 0.0001 * action_sum;

  return reward;
}

Environment create_swimmer_env(){
  return create_mujoco_env("./assets/swimmer.xml", step, 2);
}



