/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <MLP.h>
#include <mnist.h>
#include <string.h>


int main(void){
  srand(time(NULL));

	MLP n = createMLP(28*28, 15, 10);
	n.plasticity = 0.2; //0.2 is a good starting learning rate - generally, the more layers/neurons, the lower your learning rate should be.

	size_t epochs = 30;
	int epoch = 0;

	//Training data
	{
		printf("Training for %lu epochs.\n", epochs);
		ImageSet training_set;
		size_t height, width;
		openImageSet(&training_set, 47040016, "../mnist/train-images-idx3-ubyte", "../mnist/train-labels-idx1-ubyte"); //You may need to provide your own mnist file
		float avgCost = 0;

		for(size_t i = 0; i < training_set.numImages * epochs; i++){ //Run for the given number of epochs
			size_t index = i % training_set.numImages;
			float* img = img2floatArray(&training_set, index, &height, &width); //Image is returned as a float array (must have same dims as input layer)
			setInputs(&n, img); //Set the activations of the neurons in the input layer.

			int correctlabel = label(&training_set, index); //Retrieve the label from the image set.

			float c = descend(&n, correctlabel); //Perform feedforward & backpropagation, and return cost

			avgCost += c;

			//Save the state of the network at the end of each epoch.
			if(i % training_set.numImages == 0 && i != 0){
				printf("Epoch %d finished, cost %f.\n", epoch++, avgCost/i);
				saveMLPToFile(&n, "../saves/mnist_784_25_10.mlp");
			}
		}
	}

	//Testing data
	{
		printf("Testing:\n");
		ImageSet testing_set;

		openImageSet(&testing_set, 7840016, "../mnist/t10k-images-idx3-ubyte", "../mnist/t10k-labels-idx1-ubyte");

		float avgCost = 0;
		float avgCorrect = 0;

		for(int i = 0; i < testing_set.numImages; i++){
			size_t height, width;
			float* img = img2floatArray(&testing_set, i, &height, &width); //Get image as float array
			setInputs(&n, img); //Set the activations in the input layer

			feedforward(&n); //Perform feedforward only, no backprop

			int guess = bestGuess(&n); //Get network's best guess

			avgCorrect += (guess == label(&testing_set, i));
		}
		printf("resulting avg correct: %f\n", avgCorrect/testing_set.numImages);
	}
}
