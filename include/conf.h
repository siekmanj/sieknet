#ifndef SIEKNETCONF_H
#define SIEKNETCONF_H

/*<<KERNEL START>>*/

/*
 * Print debug information during execution.
 */
//#define SIEKNET_DEBUG

/*
 * In noisy datasets, it can be necessary to clip gradients to avoid destructive parameter
 * updates. SIEKNET_MAX_GRAD will bound the maximum gradient wrt any parameter to the below
 * value.
 */
#define SIEKNET_MAX_GRAD 0.5f //the max gradient of any parameter across all network types

/*
 * LSTM's can sometimes suffer from the exploding gradient problem. In my experience,
 * this is due to an exploding cell state, which gets multiplied into the gradient. You
 * can clip the maximum cell state magnitude below, or comment it out to stop clipping
 * from occuring.
 */
#define SIEKNET_MAX_STATE 15.00f

/*
 * How many timesteps to allocate memory for. Because of the difficulties in
 * dynamically allocating memory, a static allocation takes place during
 * initialization. You can change SIEKNET_MAX_UNROLL_LENGTH to raise or lower
 * the number of unrolls recurrent networks will do during backpropagation
 * through time. The seq_len attribute of the network must be strictly less
 * than SIEKNET_MAX_UNROLL_LENGTH.
 */
#define SIEKNET_MAX_UNROLL_LENGTH    100

/*
 * Buffers that are not written to are passed in as __constant, significantly 
 * boosting performance. However, on nvidia GPU's this may sometimes result in 
 * a CL_INVALID_KERNEL_ARGS error if the buffer is too large. This is because 
 * nvidia's OpenCL implementation caches __constant memory.
 */
#define SIEKNET_AMDGPU_READONLY_SPEEDUP 

/*
 * Writing data to the GPU is the cause of a non-trivial amount of overhead.
 * On some systems, passing a single argument and using a kernel to construct
 * a one-hot vector on the GPU instead of writing an entire one-hot vector from
 * the host may be more efficient. On my system, it doesn't make much difference.
 */
//#define SIEKNET_ONEHOT_SPEEDUP

/*
 * Reading data from the GPU is the cause of a non-trivial amount of overhead.
 * By default, the output of a neural network is read from the GPU into the host
 * machine. If this is not necessary (i.e., you are doing training and don't need
 * to see output), you can disable this reading by enabling SIEKNET_GPU_NO_OUTPUT 
 * so that no output is read from the GPU.
 */
//#define SIEKNET_GPU_NO_OUTPUT

/*
 * I'm not sure if it's possible to have more than one OpenCL platform on a machine,
 * but the OpenCL implementation allows for it. If for some reason you'd like to use
 * another platform, you can try setting SIEKNET_USE_PLATFORM to some other number
 * besides zero.
 */
#define SIEKNET_USE_PLATFORM 0

/* 
 * You may have more than one GPU device available. You can change SIEKNET_USE_DEVICE
 * to use a different GPU than just the first one available.
 */
#define SIEKNET_USE_DEVICE   0


/*
 * If enabled, stops execution upon encountering a NaN in network output. Not guaranteed
 * to work on GPU.
 */
//#define SIEKNET_STOP_ON_NAN //stop execution if any nan's are found


/*<<KERNEL END>>*/
#define throw_err(s) printf("%s\n", s); exit(1);

#endif

