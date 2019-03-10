/* Jonah Siekmann
 * 1/20/2019
 * This is a toy problem - training an mlp to convert a 4-bit binary string to a decimal number between 0 and 15.
 */

#include <lstm.h>
#include <optimizer.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;
#define PRINTLIST(name, len) printf("printing %s: [", #name); for(int xyz = 0; xyz < len; xyz++){printf("%5.4f", name[xyz]); if(xyz < len-1) printf(", "); else printf("]\n");}

float uniform(float minimum, float maximum){
	float center = minimum + (maximum - minimum)/2;
	float max_mag = maximum - center;
	if(rand()&1)
		return center + ((((float)rand())/RAND_MAX)) * max_mag;
	else
		return center - ((((float)rand())/RAND_MAX)) * max_mag;
}
int main(){
	srand(1);
	int input_dim = 4;
	int trials = 50;

	LSTM n = create_lstm(input_dim, 2, 4);
	SGD o = create_optimizer(SGD, n);
	n.seq_len = 4;


	float avg_time = 0;
	

	float x[input_dim];
	for(int i = 0; i < input_dim; i++)
		x[i] = uniform(-1, 1);

	for(int i = 0; i < trials; i++){
    clock_t start = clock();
		lstm_forward(&n, x);
		lstm_cost(&n, x);
		lstm_backward(&n);
		o.step(o);
    avg_time += ((float)(clock() - start)) / CLOCKS_PER_SEC;
	}
#ifdef GPU
	float *tmp = (float*)malloc(sizeof(float) * n.num_params);
	clEnqueueReadBuffer(get_opencl_queue(), n.gpu_params, 1, 0, sizeof(float) * n.num_params, n.params, 0, NULL, NULL);
#endif
	for(int i = 0; i < n.num_params; i++){
		printf("param %d: %f\n", i, n.params[i]);
	}
	printf("avg time over %d trials: %f\n", trials, avg_time);

	PRINTLIST(n.output, n.output_dimension);

	printf("exit!\n");
}
