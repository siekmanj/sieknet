/* Jonah Siekmann
 * 7/24/2018
 * Loads a .mlp file from /saves/ and runs it on the mnist test set
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <MLP.h>
#include <mnist.h>

const int RANGE = 7;

void loadtest(){
	MLP n = loadMLPFromFile("../saves/mnist_trained.mlp");
	//Testing data
  {
		printf("Testing:\n");
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
}


int main(){
	srand(time(NULL));
	//binarySolver();
	//mnist();
	loadtest();
}
