#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "./net/network.h"
#include "./mnist/mnist.h"

const int RANGE = 7;
int main(){
	ImageSet training_set;
	openImageSet(&training_set, 7840016, "./mnist/t10k-images-idx3-ubyte", "./mnist/t10k-labels-idx1-ubyte");

	srand(time(NULL));
	size_t input_layer_size = 28*28;
	size_t hidden_layer_size = 14;
	size_t output_layer_size = 10;

	Layer *input_layer = create_layer(input_layer_size, 0);
	Layer *hidden_layer = create_layer(hidden_layer_size, input_layer);
	Layer *hidden_layer2 = create_layer(hidden_layer_size, hidden_layer);
	Layer *output_layer = create_layer(output_layer_size, hidden_layer2);

	size_t height, width;
	int index = 563;
	float* img = img2floatArray(&training_set, index, &height, &width);
	setOutputs(input_layer, img);
	printf("Input image: \n");
	printImage(&training_set, index);
	printf("%d\n", label(&training_set, index));
	float c;
	for(int i = 0; i < 10000; i++){
		calculateOutputs(hidden_layer);
		calculateOutputs(hidden_layer2);
		calculateOutputs(output_layer);
		c = cost(output_layer, label(&training_set, index));
		backpropagate(output_layer);
	}
	printOutputs(output_layer);
	printf("cost: %f\n", c);


}
