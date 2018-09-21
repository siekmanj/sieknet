#include <stdio.h>
#include <math.h>
#include <MLP.h>

/*
 * This is an example problem that demonstrates a very simple use of a multilayer perceptron.
 * The network is trained to convert a 4-bit binary string into a decimal number.
 * For example, the bitstring 0 0 1 1 would result in neuron 3 of the output layer firing.
 */

int main(){
	srand(time(NULL));
	MLP n = createMLP(4, 8, 16); //Create a network with a 4-neuron input layer, 8-neuron hidden layer, and 16-neuron output layer.
	
	for(int i = 0; i < 80000000; i++){ //Run the network for 80000....00 examples
		//Create a random 4-bit binary number
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		int bit3 = rand()%2==0;
		float ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float arr[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)
		setInputs(&n, arr);

		float cost = descend(&n, (int)ans); //Calculate outputs and run backprop

		//Debug stuff
		if(!(i % 1000)){
			printf("CURRENTLY ON EXAMPLE %d\n", i);
			printOutputs(n.output);
			printWeights(n.output);
			printf("Label %2d, guess %2d, Cost: %5.3f\n\n(ENTER to continue, CTRL+C to quit)\n", (int)ans, bestGuess(&n), cost);
			getchar();
		}	
	}
}
