#include <opencl_utils.h>
#ifdef GPU
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define KERNEL_START_SYMBOL "/*<<KERNEL START>>*/\n"
#define KERNEL_END_SYMBOL   "/*<<KERNEL END>>*/\n"

void check_error(int err, char *str){
  if(err != CL_SUCCESS){
    printf("ERROR: '%s': ", str);
    switch(err){
      case CL_INVALID_PROGRAM:
        printf("CL_INVALID_PROGRAM.\n");
        break;
      case CL_INVALID_PROGRAM_EXECUTABLE:
        printf("CL_INVALID_PROGRAM_EXECUTABLE.\n");
        break;
      case CL_INVALID_KERNEL_NAME:
        printf("CL_INVALID_KERNEL_NAME.\n");
        break;
      case CL_INVALID_KERNEL_DEFINITION:
        printf("CL_INVALID_KERNEL_DEFINITION.\n");
        break;
      case CL_INVALID_VALUE:
        printf("CL_INVALID_VALUE.\n");
        break;
      case CL_OUT_OF_HOST_MEMORY:
        printf("CL_OUT_OF_HOST_MEMORY.\n");
        break;
      case CL_INVALID_COMMAND_QUEUE:
        printf("CL_INVALID_COMMAND_QUEUE.\n");
        break;
      case CL_INVALID_KERNEL:
        printf("CL_INVALID_KERNEL.\n");
        break;
			case CL_INVALID_ARG_INDEX:
				printf("CL_INVALID_ARG_INDEX.\n");
				break;
			case CL_INVALID_ARG_VALUE:
				printf("CL_INVALID_ARG_VALUE.\n");
				break;
			case CL_INVALID_MEM_OBJECT:
				printf("CL_INVALID_MEM_OBJECT.\n");
				break;
			case CL_INVALID_ARG_SIZE:
				printf("CL_INVALID_ARG_SIZE.\n");
				break;
      case CL_INVALID_CONTEXT:
        printf("CL_INVALID_CONTEXT.\n");
        break;
      case CL_INVALID_KERNEL_ARGS:
        printf("CL_INVALID_KERNEL_ARGS.\n");
        break;
      case CL_INVALID_WORK_DIMENSION:
        printf("CL_INVALID_WORK_DIMENSION.\n");
        break;
      case CL_INVALID_WORK_GROUP_SIZE:
        printf("CL_INVALID_WORK_GROUP_SIZE.\n");
        break;
      case CL_INVALID_WORK_ITEM_SIZE:
        printf("CL_INVALID_WORK_ITEM_SIZE.\n");
        break;
      case CL_INVALID_GLOBAL_OFFSET:
        printf("CL_INVALID_GLOBAL_OFFSET.\n");
        break;
			case CL_INVALID_DEVICE:
				printf("CL_INVALID_DEVICE.\n");
				break;
			case CL_INVALID_BINARY:
				printf("CL_INVALID_BINARY.\n");
				break;
			case CL_INVALID_BUILD_OPTIONS:
				printf("CL_INVALID_BUILD_OPTIONS.\n");
				break;
			case CL_INVALID_OPERATION:
				printf("CL_INVALID_OPERATION.\n");
				break;
			case CL_COMPILER_NOT_AVAILABLE:
				printf("CL_COMPILER_NOT_AVAILABLE.\n");
				break;
			case CL_BUILD_PROGRAM_FAILURE:
				printf("CL_BUILD_PROGRAM_FAILURE.\n");
				break;
      default:
        printf("default err.\n");
        break;
    }
    exit(1);
  }
}

#define USE_PLATFORM 0
#define USE_DEVICE   0

static cl_device_id GLOBAL_DEVICE;

cl_context create_opencl_context(){
	cl_uint num_platforms, num_devices;
	int status = clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platforms[num_platforms];

	check_error(clGetPlatformIDs(num_platforms, platforms, NULL), "couldn't get platforms");
	check_error(clGetDeviceIDs(platforms[USE_PLATFORM], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "couldn't count devices");
	
	cl_device_id devices[num_devices];

	check_error(clGetDeviceIDs(platforms[USE_PLATFORM], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL), "couldn't get device ids.");

	const cl_context_properties cfg[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[USE_PLATFORM], 0, 0};

	cl_context context = clCreateContext(cfg, num_devices, devices, NULL, NULL, &status);

	check_error(status, "couldn't make context");
	printf("successfully created opencl context.\n");
	return context;
}

cl_device_id get_device(){
	cl_uint num_platforms, num_devices;
	int status = clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platforms[num_platforms];

	check_error(clGetPlatformIDs(num_platforms, platforms, NULL), "couldn't get platform ids");
	check_error(clGetDeviceIDs(platforms[USE_PLATFORM], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "couldn't count gpu device ids");
	
	cl_device_id devices[num_devices];

	check_error(clGetDeviceIDs(platforms[USE_PLATFORM], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL), "couldn't get gpu devices");
	return devices[USE_DEVICE];
}

cl_command_queue make_opencl_queue(cl_context c){
	cl_device_id device = get_device();

	int err;
	cl_command_queue queue = clCreateCommandQueue(c, device, 0, &err);
	check_error(err, "couldn't create command queue");

	return queue;
}

cl_program build_program(cl_context context, cl_device_id device, char *source_code){
	int err;
	cl_program prog = clCreateProgramWithSource(context, 1, (const char**)&source_code, NULL, &err);
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

	size_t len = 0;
	check_error(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len), "getting length of compiler output");

	char buffer[len];
	check_error(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL), "copying compiler output to buffer");

	printf("<OPENCL COMPILER OUTPUT:\n%s<END OPENCL COMPILER OUTPUT>\n", buffer);
	check_error(err, "couldn't build program");
	return prog;
}

char *get_kernel_source(char **filenames, size_t numfiles){
	//get the max cumulative filelen
	size_t kernelfilelen = 0;
	for(int i = 0; i < numfiles; i++){
		FILE *fp = fopen(filenames[i], "rb");
		if(!fp){
			printf("couldn't find '%s'.\n", filenames[i]);
			exit(1);
		}
		fseek(fp, 0, SEEK_END);
		kernelfilelen += ftell(fp);
		fclose(fp);
	}
	char *ret = (char*)calloc(kernelfilelen+1, sizeof(char));
	for(int i = 0; i < numfiles; i++){
		printf("doing file '%s'\n", filenames[i]);
		FILE *fp = fopen(filenames[i], "rb");
		char current_line[5000];
		do{
			if(!fgets(current_line, 5000, fp)){
				throw_err("unable to read kernel file, missing kernel start marker.\n");
			}
		}
		while(strcmp(KERNEL_START_SYMBOL, current_line));
		printf("looking for end token\n");

		do{
			if(!fgets(current_line, 1000, fp)){
				throw_err("unable to read nonlinearity file, missing kernel end marker.\n");
			}

			//the static keyword is disallowed by OpenCL 1.1 (which my gpu uses) so we need to remove it from the kernels.
			char static_check[8];
			memset(static_check, '\0', 8);
			int x = 6;
			while(x --> 0)
				static_check[x] = current_line[x];

			if(strcmp(KERNEL_END_SYMBOL, current_line)){
				if(!strcmp("static", static_check))
					strcat(ret, current_line + strlen("static")+1);
				else
					strcat(ret, current_line);
			}
		}
		while(strcmp(KERNEL_END_SYMBOL, current_line));
		//fclose(fp);
	}
	return ret;
}
#endif
