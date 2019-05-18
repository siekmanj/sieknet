#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mj_env.h>

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

    tmp->window = glfwCreateWindow(1200, 900, "MuJoCo", NULL, NULL);
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
  tmp->camera.distance = 5;
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

Environment create_mujoco_env(char *xml, float (*step)(Environment, float *), int qpos_start){
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
  d->model = mj_loadXML(xml, 0, error, 1000);
  if(!d->model)
    mju_error_s("Load model error: %s", error);

  d->data = mj_makeData(d->model);
  d->render_setup = 0;
  d->qpos_start = qpos_start;

  env.observation_space = d->model->nq + d->model->nv - qpos_start;
  env.action_space = d->model->nu;

  env.data = d;
  env.state = (float*)calloc(env.observation_space, sizeof(float));

  env.done = calloc(1, sizeof(int));
  return env;
}

