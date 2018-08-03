#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <MLP.h>
#include <mnist.h>

//Toy problem - trains an MLP to convert from binary to decimal
int main(){
  srand(time(NULL));
	MLP n = initMLP();
	addLayer(&n, 4);
	addLayer(&n, 8);
	addLayer(&n, 16);

	for(int i = 0; i < 8000; i++){
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
		if(i % 500 == 0){
			printOutputs(n.output);
			printf("Label %f, Cost: %f\n\n\n", ans, cost);

		}
	}
}
