#include <opencl_utils.h>
#include <conf.h>
#ifdef SIEKNET_USE_GPU
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define KERNEL_START_SYMBOL "/*<<KERNEL START>>*/\n"
#define KERNEL_END_SYMBOL   "/*<<KERNEL END>>*/\n"

#ifndef SIEKNET_USE_DEVICE
#define SIEKNET_USE_DEVICE 0
#endif

#ifndef SIEKNET_USE_PLATFORM
#define SIEKNET_USE_PLATFORM 0
#endif

static int SIEKNET_IS_GPU_INITIALIZED = 0;
static cl_context SIEKNET_CONTEXT = NULL;
static cl_command_queue SIEKNET_QUEUE0 = NULL;
static cl_command_queue SIEKNET_QUEUE1 = NULL;
static cl_command_queue SIEKNET_QUEUE2 = NULL;
static cl_command_queue SIEKNET_QUEUE3 = NULL;
static cl_device_id SIEKNET_DEVICE = NULL;
static cl_mem SIEKNET_SMSUM = NULL;
static cl_mem SIEKNET_COSTSCL = NULL;

/*
 * This function checks 'err' for OpenCL error codes,
 * and prints the error code if found.
 */
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


/*
 * Creates an OpenCL context object and returns it.
 * Global variable SIEKNET_USE_PLATFORM is 0, no guarantees
 * if you use any other value than 0.
 */
static cl_context create_opencl_context(){
	cl_uint num_platforms, num_devices;
	int status = clGetPlatformIDs(0, NULL, &num_platforms);
	if(num_platforms == 0){
		printf("ERROR: create_opencl_context(): no platforms found. OpenCL was not able to find a GPU, or was not correctly installed.\n");
		exit(1);
	}
	if(SIEKNET_USE_PLATFORM >= num_platforms){
		printf("ERROR: create_opencl_context(): USE PLATFORM is %d, but only platforms 0 through %d available.\n", SIEKNET_USE_PLATFORM, num_platforms-1);
		exit(1);
	}

	cl_platform_id platforms[num_platforms];

	check_error(clGetPlatformIDs(num_platforms, platforms, NULL), "couldn't get platforms");
	check_error(clGetDeviceIDs(platforms[SIEKNET_USE_PLATFORM], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "couldn't count devices");
	
	cl_device_id devices[num_devices];

	check_error(clGetDeviceIDs(platforms[SIEKNET_USE_PLATFORM], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL), "couldn't get device ids.");

	const cl_context_properties cfg[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[SIEKNET_USE_PLATFORM], 0, 0};

	cl_context context = clCreateContext(cfg, num_devices, devices, NULL, NULL, &status);

	check_error(status, "couldn't make context");
#ifdef SIEKNET_DEBUG
	/* prints debug information about GPU driver if the SIEKNET_DEBUG flag in conf.h is set */
	printf("\nOPENCL_SETUP: GPU platform information:\n");
	size_t size;
	char *str;
	check_error(clGetPlatformInfo(platforms[SIEKNET_USE_PLATFORM], CL_PLATFORM_NAME, 0, NULL, &size), "couldn't retrieve platform name size");
	str = (char*)malloc(sizeof(char)*size);
	check_error(clGetPlatformInfo(platforms[SIEKNET_USE_PLATFORM], CL_PLATFORM_NAME, size, str, NULL), "couldn't retrieve platform name");
	printf("OPENCL_SETUP: \tPlatform name: '%s'\n", str);
	free(str);
	check_error(clGetPlatformInfo(platforms[SIEKNET_USE_PLATFORM], CL_PLATFORM_VENDOR, 0, NULL, &size), "couldn't retrieve platform vendor name size");
	str = (char*)malloc(sizeof(char)*size);
	check_error(clGetPlatformInfo(platforms[SIEKNET_USE_PLATFORM], CL_PLATFORM_VENDOR, size, str, NULL), "couldn't retrieve platform vendor");
	printf("OPENCL_SETUP: \tPlatform vendor: '%s'\n", str);
	free(str);
	check_error(clGetPlatformInfo(platforms[SIEKNET_USE_PLATFORM], CL_PLATFORM_VERSION, 0, NULL, &size), "couldn't retrieve platform vendor name size");
	str = (char*)malloc(sizeof(char)*size);
	check_error(clGetPlatformInfo(platforms[SIEKNET_USE_PLATFORM], CL_PLATFORM_VERSION, size, str, NULL), "couldn't retrieve platform vendor");
	printf("OPENCL_SETUP: \tPlatform version: '%s'\n", str);
	free(str);

#endif
	return context;
}

/*
 * Creates an OpenCL device ID object using globally defined
 * USE_DEVICE (by default set to 0, first available GPU). No
 * guarantees if you use any other value.
 */
cl_device_id create_opencl_device(){
	cl_uint num_platforms, num_devices;
	int err = clGetPlatformIDs(0, NULL, &num_platforms);
	check_error(err, "getting opencl platforms");

	cl_platform_id platforms[num_platforms];

	check_error(clGetPlatformIDs(num_platforms, platforms, NULL), "couldn't get platform ids");
	check_error(clGetDeviceIDs(platforms[SIEKNET_USE_PLATFORM], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices), "couldn't count gpu device ids");
	if(SIEKNET_USE_DEVICE >= num_devices){
		printf("ERROR: create_opencl_device(): SIEKNET_USE_DEVICE is %d, but only devices 0 through %d available.\n", SIEKNET_USE_DEVICE, num_devices-1);
		exit(1);
	}
	
	cl_device_id devices[num_devices];

	check_error(clGetDeviceIDs(platforms[SIEKNET_USE_PLATFORM], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL), "couldn't get gpu devices");
#ifdef SIEKNET_DEBUG
	/* prints debug information about GPU if the SIEKNET_DEBUG flag in conf.h is set */
	cl_uint ui;
	char *str;
	size_t size;
	size_t sizes[3] = {0, 0, 0};
	printf("\nOPENCL_SETUP: GPU device information:\n");
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_NAME, 0, NULL, &size);
	str = (char*)malloc(sizeof(char)*size);
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_NAME, size, str, NULL);
	printf("OPENCL_SETUP: \tDevice Name = %s\n", str);
	free(str);
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_NAME, 0, NULL, &size);
	printf("OPENCL_SETUP: \tDevice Vendor ID = 0x%04x\n", ui );
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_VENDOR_ID, sizeof(ui), &ui, NULL );
	printf("OPENCL_SETUP: \tDevice Vendor ID = 0x%04x\n", ui );
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui), &ui, NULL );
	printf("OPENCL_SETUP: \tDevice Maximum Compute Units = %d\n", ui );
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(ui), &ui, NULL );
	printf("OPENCL_SETUP: \tDevice Maximum Work Item Dimensions = %d\n", ui );
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(sizes), sizes, NULL );
	printf("OPENCL_SETUP: \tDevice Maximum Work Item Sizes = %lu x %lu x %lu\n", sizes[0], sizes[1], sizes[2] );
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL );
	printf("OPENCL_SETUP: \tDevice Maximum Work Group Size = %lu\n", size );
	clGetDeviceInfo(devices[SIEKNET_USE_DEVICE], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ui), &ui, NULL );
	printf("OPENCL_SETUP: \tDevice Maximum Clock Frequency = %d MHz\n", ui );
#endif
	return devices[SIEKNET_USE_DEVICE];
}

/*
 * Creates an OpenCL command queue object.
 */
static cl_command_queue create_opencl_queue(cl_context c){
	int err;
	cl_command_queue queue = clCreateCommandQueue(c, SIEKNET_DEVICE, 0, &err);
	check_error(err, "couldn't create command queue");

	return queue;
}

/* 
 * Creates and attempts to build an OpenCL program given a cstring 
 * of source code.
 */
cl_program build_program(char *source_code){
	int err;
	//printf("OPENCL_SETUP: building src:\n%s\n", source_code);
	cl_program prog = clCreateProgramWithSource(SIEKNET_CONTEXT, 1, (const char**)&source_code, NULL, &err);
	check_error(err, "couldn't create program from source");
	clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	check_error(err, "couldn't build program");

	size_t len = 0;
	check_error(clGetProgramBuildInfo(prog, SIEKNET_DEVICE, CL_PROGRAM_BUILD_LOG, 0, NULL, &len), "getting length of compiler output");

	char buffer[len];
	check_error(clGetProgramBuildInfo(prog, SIEKNET_DEVICE, CL_PROGRAM_BUILD_LOG, len, buffer, NULL), "copying compiler output to buffer");

#ifdef SIEKNET_DEBUG
  /* prints the opencl compiler output if any warnings or errors are present and SIEKNET_DEBUG is set */
	if(strlen(buffer) > 0)
		printf("\n<OPENCL COMPILER WARNINGS & ERRORS>:\n%s<END OPENCL COMPILER OUTPUT>\n\n", buffer);
#endif
	return prog;
}

/* 
 * Initializes OpenCL objects 
 */
void initialize_opencl(){
	/* initialize_opencl can be called by many different functions, so we need to *
	 * check that it hasn't already been called (only needs to be run one)        */
	if(!SIEKNET_IS_GPU_INITIALIZED){
		int err;
		SIEKNET_CONTEXT = create_opencl_context();
		SIEKNET_DEVICE  = create_opencl_device();
		SIEKNET_QUEUE0  = create_opencl_queue(SIEKNET_CONTEXT);
		SIEKNET_QUEUE1  = create_opencl_queue(SIEKNET_CONTEXT);
		SIEKNET_QUEUE2  = create_opencl_queue(SIEKNET_CONTEXT);
		SIEKNET_QUEUE3  = create_opencl_queue(SIEKNET_CONTEXT);
		SIEKNET_SMSUM   = clCreateBuffer(SIEKNET_CONTEXT, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
		SIEKNET_COSTSCL = clCreateBuffer(SIEKNET_CONTEXT, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
		check_error(err, "failed to create softmax-sum placeholder");
	}
	SIEKNET_IS_GPU_INITIALIZED = 1;
}

/* Getters for returning various OpenCL objects. Used to prevent modification. */
cl_context get_opencl_context(){
	return SIEKNET_CONTEXT;
}
cl_device_id get_opencl_device(){
	return SIEKNET_DEVICE;
}
cl_command_queue get_opencl_queue0(){
	return SIEKNET_QUEUE0;
}
cl_command_queue get_opencl_queue1(){
	return SIEKNET_QUEUE1;
}
cl_command_queue get_opencl_queue2(){
	return SIEKNET_QUEUE2;
}
cl_command_queue get_opencl_queue3(){
	return SIEKNET_QUEUE3;
}
cl_mem get_softmax_sum(){
	return SIEKNET_SMSUM;
}
cl_mem get_cost_scalar(){
	return SIEKNET_COSTSCL;
}

/* 
 * Builds a cstring of kernel source code from a list of kernel filenames.
 * Each file is inspected for the KERNEL_START_SYMBOL (defined at the top
 * of this file), whereupon the source code is read into the buffer. Upon seeing
 * a KERNEL_END_SYMBOL, the function stops appending code from that file and 
 * moves on to the next one.
 */
char *get_kernel_source(char **filenames, size_t numfiles){

	/* get the sum of filelengths across all files (final buffer can't be bigger than that) */
	size_t kernelfilelen = 0;
	for(int i = 0; i < numfiles; i++){
		FILE *fp = fopen(filenames[i], "rb");
		if(!fp){
			printf("OPENCL_SETUP: couldn't find '%s'.\n", filenames[i]);
			exit(1);
		}
		fseek(fp, 0, SEEK_END);
		kernelfilelen += ftell(fp);
		fclose(fp);
	}

	/* allocate the space for the buffer */
	char *ret = (char*)calloc(kernelfilelen+1, sizeof(char));

	/* begin reading source files for kernels */
	for(int i = 0; i < numfiles; i++){
		FILE *fp = fopen(filenames[i], "rb");
		char current_line[5000];
		do{
			/* if we get to the end of the file without seeing a kernel start marker, exit */
			if(!fgets(current_line, 5000, fp)){
				throw_err("unable to read kernel file, missing kernel start marker.\n");
			}
		}
		while(strcmp(KERNEL_START_SYMBOL, current_line));

		do{
			/* if we get to the end of a kernel file without seeing the kernel end marker, exit */
			if(!fgets(current_line, 5000, fp)){
				throw_err("unable to read kernel file, missing kernel end marker.\n");
			}

			/* the static keyword is disallowed by OpenCL 1.1 (which my gpu uses) *
			 * the following lines remove 'static' from strings if found.         */
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

		//sometimes causes segfaults - may be an issue with earlier memory usage?
		fclose(fp);
	}
	return ret;
}
#endif
