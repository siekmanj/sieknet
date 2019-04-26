#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hopper2d_env.h>
#include <mujoco.h>
#include <glfw3.h>

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

/*
// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}
*/
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
}

static void render(Environment env){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;


  if(!tmp->render_setup){
    glfwInit();

    tmp->window = glfwCreateWindow(1200, 900, "Hopper2d", NULL, NULL);
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
  mjData *d = tmp->data;
  mjModel *m = tmp->model;

  glfwDestroyWindow(tmp->window);
  glfwTerminate();

  tmp->render_setup = 0;

}

static float step(Environment env, float *action){
  Data *tmp = ((Data*)env.data);
  mjData *d = tmp->data;
  mjModel *m = tmp->model;

  mjtNum simstart = d->time;
  while(d->time - simstart < 1.0/60.0)
    mj_step(m, d);


}

Environment create_hopper2d_env(){

  // activate software
  mj_activate("/home/jonah/.mujoco/mjkey.txt");

  Environment env;
  env.dispose = dispose;
  env.reset = reset;
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

  env.observation_space = d->model->nq;
  env.action_space = d->model->nu;

  env.data = d;

  return env;
}
