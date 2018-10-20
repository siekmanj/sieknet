/* Jonah Siekmann
 * 7/24/2018
 */

#include <stdio.h>
#include <MLP.h>
#include <mnist.h>

/*
 * This is a simple example of how to use the provided loadMLPFromFile function to
 * load a .mlp file and run it on mnist.
 */

void main(){
	srand(time(NULL));
	printf("Testing:\n");
	MLP n = loadMLPFromFile("../saves/mnist_784_20_20_10.mlp");

	ImageSet testing_set;
	openImageSet(&testing_set, 7840016, "../mnist/t10k-images-idx3-ubyte", "../mnist/t10k-labels-idx1-ubyte");

	float avgCost = 0;
	float avgCorrect = 0;

	for(int i = 0; i < testing_set.numImages; i++){
		size_t height, width;
		float *img = img2floatArray(&testing_set, i, &height, &width);

		setInputs(&n, img);
		feedforward(&n);

		int guess = bestGuess(&n);
		avgCorrect += guess == label(&testing_set, i);
	}
	printf("resulting avg correct: %f\n", avgCorrect/testing_set.numImages);
}
