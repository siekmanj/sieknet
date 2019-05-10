#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <conf.h>
#include <hopper_env.h>
#include <mujoco.h>
#include <glfw3.h>

#define ALIVE_BONUS 0.2f
#define FRAMESKIP 10

typedef struct data {
  mjModel *model;
  mjData *data;
  mjvCamera camera;
  mjvOption opt;
  mjvScene scene;
  mjrContext context;
  GLFWwindow *window;
  int render_setup;
} Data;

static float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

static float normal(float mean, float std){
	float u1 = uniform(0, 1);
	float u2 = uniform(0, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * 3.14159 * u2);
	return mean + norm * std;
}

static void dispose(Environment env){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;

  mjv_freeScene(&tmp->scene);
  mjr_freeContext(&tmp->context);
  mj_deleteData(d);
  mj_deleteModel(m);
  mj_deactivate();
}

static void reset(Environment env){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;

  mj_resetData(m, d);
  mj_forward(m, d);

  *env.done = 0;
}

static void seed(Environment env){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;

  for(int i = 0; i < m->nq; i++)
    d->qpos[i] += normal(0, 0.005);

  for(int i = 0; i < m->nu; i++)
    d->qvel[i] += normal(0, 0.005);

}

static void render(Environment env){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;


  if(!tmp->render_setup){

    tmp->window = glfwCreateWindow(1200, 900, "Hopper", NULL, NULL);
    glfwMakeContextCurrent(tmp->window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&tmp->camera);
    mjv_defaultOption(&tmp->opt);
    mjv_defaultScene(&tmp->scene);
    mjr_defaultContext(&tmp->context);

    mjv_makeScene(tmp->model, &tmp->scene, 2000);
    mjr_makeContext(tmp->model, &tmp->context, mjFONTSCALE_150);
    tmp->render_setup = 1;
  }

  tmp->camera.lookat[0] = d->qpos[0];
  tmp->camera.distance = 4.0;
  tmp->camera.elevation = -20.0;

  // get framebuffer viewport
  mjrRect viewport = {0, 0, 0, 0};
  glfwGetFramebufferSize(tmp->window, &viewport.width, &viewport.height);

  // update scene and render
  mjv_updateScene(m, d, &tmp->opt, NULL, &tmp->camera, mjCAT_ALL, &tmp->scene);
  mjr_render(viewport, &tmp->scene, &tmp->context);

  // swap OpenGL buffers (blocking call due to v-sync)
  glfwSwapBuffers(tmp->window);

  // process pending GUI events, call GLFW callbacks
  glfwPollEvents();

}

static void close(Environment env){
  Data *tmp = ((Data*)env.data);
  glfwDestroyWindow(tmp->window);

  tmp->render_setup = 0;
}

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
  Environment env;
  glfwInit();

  // activate software
  mj_activate(SIEKNET_MJKEYPATH);

  env.dispose = dispose;
  env.reset = reset;
  env.seed = seed;
  env.render = render;
  env.close = close;
  env.step = step;

  Data *d = (Data*)malloc(sizeof(Data));
  char error[1000] = "Couldn't load model file.";
  d->model = mj_loadXML("./assets/hopper.xml", 0, error, 1000);
  if(!d->model)
    mju_error_s("Load model error: %s", error);

  d->data = mj_makeData(d->model);
  d->render_setup = 0;

  env.observation_space = d->model->nq + d->model->nv - 1;
  env.action_space = d->model->nu;

  env.data = d;
  env.state = (float*)calloc(env.observation_space, sizeof(float));

  env.done = calloc(1, sizeof(int));
  return env;
}
