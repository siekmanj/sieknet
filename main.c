/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "./net/network.h"
#include "./mnist/mnist.h"

const int RANGE = 7;


void mnist(){
	Network n = initNetwork();
	addLayer(&n, 28*28); //input layer
	addLayer(&n, 16);
	addLayer(&n, 10); //output layer

	//Training data
	ImageSet training_set;
	openImageSet(&training_set, 47040016, "./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte");
	float avgCost = 0;
	for(int i = 0; i < training_set.numImages; i++){
		size_t height, width;
		float* img = img2floatArray(&training_set, i, &height, &width);
		setInputs(&n, img);

		float cost = runEpoch(&n, label(&training_set, i));
		avgCost += cost;

		if(i % 100 == 0) fprintf(stderr, ".");
		if(i % 50 == 0){
			printf("\nLabel %d, cost: %f, avgcost: %f\n\n\n", label(&training_set, i), cost, avgCost/i);
			printOutputs(n.output);
		}
	}

}

//Toy problem - trains a network to convert from binary to decimal
void binarySolver(){
	Network n = initNetwork();
	addLayer(&n, 4);
	addLayer(&n, 8);
	addLayer(&n, 16);

	for(int i = 0; i < 2000; i++){
		//Create a random 4-bit binary number
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		int bit3 = rand()%2==0;
		float ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2) + bit3 * pow(2, 3);

		float arr[4] = {bit0, bit1, bit2, bit3}; //Input array (1 bit per input)
		setInputs(&n, arr);

		float cost = runEpoch(&n, (int)ans); //Calculate outputs and run backprop

		//Debug stuff
		if(i % 500 == 0){
			printOutputs(n.output);
			printf("Label %f, Cost: %f\n\n\n", ans, cost);

		}
	}
}
int main(){
	srand(time(NULL));
	//binarySolver();
	mnist();



}
