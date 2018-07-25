#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "./net/network.h"
#include "./mnist/mnist.h"

const int RANGE = 7;


/* NOT WORKING
void mnistSolver(){
	//Get training and test datasets from MNIST folder


	//Network creation
	size_t input_layer_size = 28*28;
	size_t hidden_layer_size = 16*16;
	size_t output_layer_size = 10;

	Layer *input_layer = create_layer(input_layer_size, 0);
	Layer *hidden_layer = create_layer(hidden_layer_size, input_layer);
	//Layer *hidden_layer2 = create_layer(hidden_layer_size, hidden_layer);
	Layer *output_layer = create_layer(output_layer_size, hidden_layer);
	float avgCost = 0;
	for(int i = 0; i < training_set.numImages; i++){
		int index = i;
		size_t height, width;
		float* img = img2floatArray(&training_set, index, &height, &width);
		setOutputs(input_layer, img);
		calculateOutputs(hidden_layer, 1);
		//calculateOutputs(hidden_layer2);
		calculateOutputs(output_layer, 0);
		float c = cost(output_layer, label(&training_set, index));
		backpropagate(output_layer, 0.3, 5, label(&training_set, index));
		avgCost += c;
		//printf("Label: %d. Cost: %f.\n", label(&training_set, index), c);
		//printWeights(output_layer);
		//printActivationGradients(output_layer);

		if(!(i % 200)){
			fprintf(stderr, ".");
		}
		if(!(i % 20000)){
			printf("\nInput image: \n");
			printOutputs(input_layer, 1);
			printf("Labeled: %d, NN cost: %f\n", label(&training_set, index), c);
			printf("Hidden layer:\n");
			printOutputs(hidden_layer, 0);
			//printf("Hidden layer 2:\n");
			//printOutputs(hidden_layer2);
			printf("Output layer:\n");
			printOutputs(output_layer, 0);
		}

	}
	avgCost /= training_set.numImages;
	printf("Average cost: %f\n", avgCost);
}*/
/*
void mnist2Solver(){
	ImageSet training_set;
	openImageSet(&training_set, 47040016, "./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte");

	ImageSet test_set;
	openImageSet(&test_set, 7840016, "./mnist/t10k-images-idx3-ubyte", "./mnist/t10k-labels-idx1-ubyte");

	Network n = initNetwork();
	addLayer(n, 28*28);
	addLayer(n, 30*30);
	addLayer(n, 10);

	for(int i = 0; i < training_set.numImages; i++)



}
*/

//Toy problem
void binarySolver(){
	Network n = initNetwork();
	addLayer(&n, 3);
	addLayer(&n, 9);
	addLayer(&n, 8);

	for(int i = 0; i < 20000; i++){
		int bit0 = rand()%2==0;
		int bit1 = rand()%2==0;
		int bit2 = rand()%2==0;
		//int bit3 = rand()%2==0;
		float ans = bit0 * pow(2, 0) + bit1 * pow(2, 1) + bit2 * pow(2, 2);
		float arr[3] = {bit0, bit1, bit2};
		setInputs(&n, arr);
		float cost = runEpoch(&n, (int)ans);
		printOutputs(n.output);
	}
}
int main(){
	srand(time(NULL));
	binarySolver();
	//mnist2Solver();



}
