/* Jonah Siekmann
 * 1/20/2019
 * This is a toy problem - training an mlp to convert a 4-bit binary string to a decimal number between 0 and 15.
 */

#include <mlp.h>
#include <optimizer.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

float uniform(float minimum, float maximum){
	float center = minimum + (maximum - minimum)/2;
	float max_mag = maximum - center;
	if(rand()&1)
		return center + ((((float)rand())/RAND_MAX)) * max_mag;
	else
		return center - ((((float)rand())/RAND_MAX)) * max_mag;
}
int main(){
	gpu_setup();
	srand(time(NULL));

	MLP n = create_mlp(4, 6);

	float x[4] = {1.0, 2.0, 1.0, 0.0};
	mlp_forward(&n, x);
}
