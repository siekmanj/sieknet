#include <opencl_utils.h>
#ifdef GPU
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define KERNEL_START_SYMBOL "/*<<KERNEL START>>*/\n"
#define KERNEL_END_SYMBOL   "/*<<KERNEL END>>*/\n"

#define USE_PLATFORM 0
#define USE_DEVICE   0

static int SIEKNET_IS_GPU_INITIALIZED = 0;
static cl_context SIEKNET_CONTEXT = NULL;
static cl_command_queue SIEKNET_QUEUE = NULL;
static cl_device_id SIEKNET_DEVICE = NULL;
static cl_mem SIEKNET_SMSUM = NULL;

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
				printf("CL_INVALID_MEM_OBJECT (remember to pass in arg_value with &).\n");
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
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				printf("CL_MEM_OBJECT_ALLOCATION_FAILURE.\n");
				break;
			case CL_OUT_OF_RESOURCES:
				printf("CL_OUT_OF_RESOURCES.\n");
				break;
			case CL_INVALID_EVENT_WAIT_LIST:
				printf("CL_INVALID_EVENT_WAIT_LIST.\n");
				break;

			
      default:
        printf("default err, code %d\n", err);
        break;
    }
    exit(1);
  }
}


static cl_context create_opencl_context(){
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
	return context;
}

cl_device_id get_opencl_device(){
	cl_uint num_platforms, num_devices;
	int status = clGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id platforms[num_platforms];

	check_error(clGetPlatformIDs(num_platforms, platforms, NULL), "couldn't get platform ids");
	check_error(clGetDeviceIDs(platforms[USE_PLATFORM], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "couldn't count gpu device ids");
	
	cl_device_id devices[num_devices];

	check_error(clGetDeviceIDs(platforms[USE_PLATFORM], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL), "couldn't get gpu devices");
	return devices[USE_DEVICE];
}

static cl_command_queue create_opencl_queue(cl_context c){
	cl_device_id device = get_opencl_device();

	int err;
	cl_command_queue queue = clCreateCommandQueue(c, device, 0, &err);
	check_error(err, "couldn't create command queue");

	return queue;
}

cl_program build_program(char *source_code){
	int err;
	//printf("building src:\n%s\n", source_code);
	cl_program prog = clCreateProgramWithSource(SIEKNET_CONTEXT, 1, (const char**)&source_code, NULL, &err);
	check_error(err, "couldn't create program from source");
	clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

	size_t len = 0;
	check_error(clGetProgramBuildInfo(prog, SIEKNET_DEVICE, CL_PROGRAM_BUILD_LOG, 0, NULL, &len), "getting length of compiler output");

	char buffer[len];
	check_error(clGetProgramBuildInfo(prog, SIEKNET_DEVICE, CL_PROGRAM_BUILD_LOG, len, buffer, NULL), "copying compiler output to buffer");

#ifdef DEBUG
	printf("<OPENCL COMPILER OUTPUT:\n%s<END OPENCL COMPILER OUTPUT>\n", buffer);
#endif
	check_error(err, "couldn't build program");
	return prog;
}

void initialize_opencl(){
	if(!SIEKNET_IS_GPU_INITIALIZED){
		int err;
		SIEKNET_CONTEXT = create_opencl_context();
		SIEKNET_QUEUE   = create_opencl_queue(SIEKNET_CONTEXT);
		SIEKNET_DEVICE  = get_opencl_device();
		SIEKNET_SMSUM   = clCreateBuffer(SIEKNET_CONTEXT, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
		check_error(err, "failed to create softmax-sum placeholder");
	}
	SIEKNET_IS_GPU_INITIALIZED = 1;
}

cl_context get_opencl_context(){
	return SIEKNET_CONTEXT;
}
cl_command_queue get_opencl_queue(){
	return SIEKNET_QUEUE;
}
cl_mem get_softmax_sum(){
	return SIEKNET_SMSUM;
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
		FILE *fp = fopen(filenames[i], "rb");
		char current_line[5000];
		do{
			if(!fgets(current_line, 5000, fp)){
				throw_err("unable to read kernel file, missing kernel start marker.\n");
			}
		}
		while(strcmp(KERNEL_START_SYMBOL, current_line));

		do{
			if(!fgets(current_line, 1000, fp)){
				throw_err("unable to read kernel file, missing kernel end marker.\n");
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
