#include <conf.h>
#ifdef GPU
#ifndef OPENCL_UTILS
#define OPENCL_UTILS

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

char *get_kernel_source(char **, size_t);
void check_error(int, char *);

cl_program build_program(char *);

cl_context get_opencl_context();
cl_command_queue get_opencl_queue();
cl_device_id get_opencl_device();
cl_mem get_softmax_sum();

void initialize_opencl();

#endif
#endif
