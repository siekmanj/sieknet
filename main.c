/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "./net/MLP.h"
#include "./mnist/mnist.h"

const int RANGE = 7;

void savetest(){
	MLP n = initMLP();
	addLayer(&n, 28*28); //input layer
	addLayer(&n, 15);
	addLayer(&n, 10); //output layer
	saveToFile(&n, "test");
}

void mnist(){
	MLP n = initMLP();
	addLayer(&n, 28*28); //input layer
	addLayer(&n, 15);
	addLayer(&n, 10); //output layer

	size_t epochs = 10;
	int epoch = 0;

	//Training data
	{
		ImageSet training_set;
		size_t height, width;
		openImageSet(&training_set, 47040016, "./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte");

		float avgCost = 0;

		for(size_t i = 0; i < training_set.numImages * epochs; i++){
			size_t index = i % training_set.numImages;
			float* img = img2floatArray(&training_set, index, &height, &width);

			setInputs(&n, img);
			float c = descend(&n, label(&training_set, index));
			avgCost += c;

			if(i % training_set.numImages == 0 && i != 0){
				printf("Epoch %d finished, cost %f.\n", epoch++, avgCost/i);
			}
		}
		//printf("Cost that batch: %f\n", batchcost);

		printf("\nAvg training cost: %f\n", avgCost / (training_set.numImages*epochs));
	}
	//Testing data
	{
		printf("TESTING:\n");
		ImageSet testing_set;
		openImageSet(&testing_set, 7840016, "./mnist/t10k-images-idx3-ubyte", "./mnist/t10k-labels-idx1-ubyte");
		float avgCost = 0;
		float avgCorrect = 0;
		for(int i = 0; i < testing_set.numImages; i++){
			size_t height, width;
			float* img = img2floatArray(&testing_set, i, &height, &width);
			setInputs(&n, img);
			feedforward(&n);
			int guess = bestGuess(&n);
			/*
			if(i % 2000 == 0 && i != 0){
				printOutputs(n.output);
			  printf("Label: %d, guess: %d\n", label(&testing_set, i), guess);
			}
			*/
			avgCorrect += guess == label(&testing_set, i);
		}
		printf("resulting avg correct: %f\n", avgCorrect/testing_set.numImages);
	}
}

//Toy problem - trains an MLP to convert from binary to decimal
void binarySolver(){
	MLP n = initMLP();
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

		float cost = descend(&n, (int)ans); //Calculate outputs and run backprop

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
	//mnist();
	savetest();



}
