#include <stdio.h>
#include <math.h>
#include <mlp.h>
#include <string.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;
/*
 * This is an example problem that demonstrates a very simple use of a multilayer perceptron.
 * The network is trained to convert a 4-bit binary string into a decimal number.
 * For example, the bitstring 0 0 1 1 would result in neuron 3 of the output layer firing.
 */

int main(){
	srand(time(NULL));
	MLP n = createMLP(2, 2, 2); //Create a network with a 4-neuron input layer, 8-neuron hidden layer, and 16-neuron output layer.
	n.learning_rate = 0.05;
	printf("num params: %lu\n", n.num_params);
	
	for(int i = 0; i < 2; i++){ //Run the network for 80000....00 examples
		//Create a random 4-bit binary number
		/*
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		int bit3 = rand()%2==0;
		float ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float x[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)
		CREATEONEHOT(y, 16, (int)ans);
		*/
		float x[] = {0.8, 0.25};
		float y[] = {0.0, 1.0};
		mlp_forward(&n, x);
		float cost = n.cost(&n, y);
		mlp_backward(&n);
		getchar();

		/*
		//Debug stuff
		if(!(i % 100)){
			printf("CURRENTLY ON EXAMPLE %d\n", i);
			printf("Label %2d, guess %2d, Cost: %5.3f\n\n(ENTER to continue, CTRL+C to quit)\n", (int)ans, n.guess, cost);
			getchar();
		}	
		*/
	}
}
