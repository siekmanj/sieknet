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
	int input_dim = 2;
	int trials = 10;

/*
	MLP m = create_mlp(input_dim, 1, input_dim);
	float x[] = {0.5, 1.0};
	float y[] = {0.0, 1.0};

	mlp_forward(&m, x);
	mlp_cost(&m, y);
	mlp_backward(&m);
*/
	LSTM n = create_lstm(input_dim, 1, input_dim);
	SGD o = create_optimizer(SGD, n);
	n.seq_len = 4;


	float avg_time = 0;
	

	float x[input_dim];
	for(int i = 0; i < input_dim; i++)
		x[i] = uniform(-1, 1);

	for(int i = 0; i < trials; i++){
    clock_t start = clock();
		printf("doing forward\n");
		lstm_forward(&n, x);
		PRINTLIST(n.output, n.output_dimension);
		printf("doing cost\n");
		lstm_cost(&n, x);
		printf("doing backward\n");
		lstm_backward(&n);
		//printf("doing optimizer step\n");
		//o.step(o);
    avg_time += ((float)(clock() - start)) / CLOCKS_PER_SEC;
	}
	printf("exit!\n");
}
