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
  
  float reward = (d->qpos[0] - posbefore) / (d->time - simstart);
  reward += env.alive_bonus;

  float action_sum = 0;
  for(int i = 0; i < env.action_space; i++)
    action_sum += action[i]*action[i];

  reward -= 1e-3 * action_sum;

  if(d->qpos[1] < 0.8 || d->qpos[1] > 2.0 || d->qpos[2] < -1.0 || d->qpos[2] > 1.0){
    *env.done = 1;
  }

  return reward;
}

Environment create_walker2d_env(){
  Environment ret = create_mujoco_env("./assets/walker2d.xml", step, 1);
  ret.alive_bonus = 1.0f;
  ret.frameskip = 4;
  return ret;
}

