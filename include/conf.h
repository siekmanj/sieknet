#ifndef SIEKNETCONF_H
#define SIEKNETCONF_H

/*<<KERNEL START>>*/

#define SIEKNET_DEBUG       //print debug information during execution
#define SIEKNET_MAX_GRAD           0.25f //the max gradient of any parameter across all network types
#define SIEKNET_MAX_STATE         50.00f //the max value for any cell state in an lstm
#define SIEKNET_MAX_UNROLL_LENGTH    100 //the max number of timesteps that can be backpropagated through (n.seq_len must be less)


//#define SIEKNET_ONEHOT_SPEEDUP //send only a single int across the PCI-E slot to GPU when using one-hot vectors.
//#define SIEKNET_GPU_NO_OUTPUT

#define SIEKNET_USE_PLATFORM 0 //use first available opencl platform
#define SIEKNET_USE_DEVICE   0 //use first available GPU

/* this is a list of compile-time options which you can
 * copy and paste above to enable

#define SIEKNET_STOP_ON_NAN //stop execution if any nan's are found


 *
 */

/*<<KERNEL END>>*/
#define throw_err(s) printf("%s\n", s); exit(1);

#endif

