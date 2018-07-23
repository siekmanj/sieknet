#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "./net/network.h"
#include "./mnist/mnist.h"

const int RANGE = 7;
int main(){
	srand(time(NULL));

	//Get training and test datasets from MNIST folder
	ImageSet training_set;
	openImageSet(&training_set, 47040016, "./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte");

	ImageSet test_set;
	openImageSet(&test_set, 7840016, "./mnist/t10k-images-idx3-ubyte", "./mnist/t10k-labels-idx1-ubyte");

	//Network creation
	size_t input_layer_size = 28*28;
	size_t hidden_layer_size = 10*10;
	size_t output_layer_size = 10;

	Layer *input_layer = create_layer(input_layer_size, 0);
	Layer *hidden_layer = create_layer(hidden_layer_size, input_layer);
	Layer *hidden_layer2 = create_layer(hidden_layer_size, hidden_layer);
	Layer *output_layer = create_layer(output_layer_size, hidden_layer2);

	for(int i = 0; i < training_set.numImages; i++){
		int index = i;
		size_t height, width;
		float* img = img2floatArray(&training_set, index, &height, &width);
		setOutputs(input_layer, img);
		calculateOutputs(hidden_layer);
		calculateOutputs(hidden_layer2);
		calculateOutputs(output_layer);
		float c = cost(output_layer, label(&training_set, index));
		backpropagate(output_layer, 1);
		if(!(i % 200)){
			fprintf(stderr, ".");
		}
		if(!(i % 5000)){
			printf("\nInput image: \n");
			printImage(&training_set, index);
			printf("Labeled: %d, NN cost: %f\n", label(&training_set, index), c);
			printOutputs(output_layer);
		}
	}


}
