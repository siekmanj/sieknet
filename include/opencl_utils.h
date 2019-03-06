#include <conf.h>
#ifdef GPU
#ifndef OPENCL_UTILS
#define OPENCL_UTILS

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

char *get_kernel_source(char **, size_t);
void check_error(int, char *);

cl_program build_program(cl_context, cl_device_id, char *);
cl_command_queue make_opencl_queue(cl_context);
cl_context create_opencl_context();
cl_device_id get_device();

#endif
#endif
