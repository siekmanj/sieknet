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

}

static void reset(Environment env){

}

static void render(Environment env){

}

static float step(Environment env, float *action){

}

Environment create_hopper2d_env(){

  // activate software
  mj_activate("mjkey.txt");

  Environment env;
  env.dispose = dispose;
  env.reset = reset;
  env.render = render;
  env.step = step;

  //env.observation_space = 0;
  //env.action_space = 0;

  Data d;
  /*
  d.window = glfwCreateWindow(1200, 900, "Hopper2d", NULL, NULL);
  glfwMakeContextCurrent(d.window);
  glfwSwapInterval(1);

  mjv_defaultCamera(&d.camera);
  mjv_defaultOption(&d.opt);
  mjv_defaultScene(&d.scene);
  mjv_defaultContext(&d.context);

  mjv_makeScene(&d
*/
  return env;

}
