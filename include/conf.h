#ifndef SIEKNETCONF_H
#define SIEKNETCONF_H

/*<<KERNEL START>>*/

#define SIEKNET_MAX_GRAD  0.5f
#define SIEKNET_MAX_STATE 10.0f
#define SIEKNET_STOP_ON_NAN
#define SIEKNET_DEBUG

#define SIEKNET_USE_PLATFORM 0 //use first available opencl platform
#define SIEKNET_USE_DEVICE   0 //use first available GPU

/* this is a list of compile-time options which you can
 * copy and paste above to enable


 *
 */

/*<<KERNEL END>>*/
#define throw_err(s) printf("%s\n", s); exit(1);

#endif

