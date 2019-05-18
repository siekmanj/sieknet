#ifndef MJ_ENV_H
#define MJ_ENV_H

#include <conf.h>
#include <mujoco.h>
#include <glfw3.h>
#include <env.h>

typedef struct data {
  mjModel *model;
  mjData *data;
  mjvCamera camera;
  mjvOption opt;
  mjvScene scene;
  mjrContext context;
  GLFWwindow *window;
  int render_setup;
  int qpos_start;
} Data;


Environment create_mujoco_env(char *, float (*step)(Environment, float *), int);

#endif
