/* Jonah Siekmann
 * 7/24/2018
 * This program trains an mlp to perform handwritten digit recognition using the standard MNIST dataset.
 * I've included my mnist data loader, but you're welcome to write your own. It's pretty straightforward.
 * You will need to provide your own mnist dataset files - they're a little large to host on github.
 * You can find the dataset at: http://yann.lecun.com/exdb/mnist/
 */

#include <mlp.h>
#include <optimizer.h>
#include <math.h>
#include <mnist.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define CREATEONEHOT(name, size, index) float name[size]; memset(name, '\0', size*sizeof(float)); name[index] = 1.0;

char *modelfile = "../model/mnist.mlp";

char *trainset_images = "../data/mnist/train-images.idx3-ubyte";
char *testset_images = "../data/mnist/t10k-images.idx3-ubyte";

char *trainset_labels = "../data/mnist/train-labels.idx1-ubyte";
char *testset_labels = "../data/mnist/t10k-labels.idx1-ubyte";

int main(void) {
	srand(time(NULL));

	//MLP n = loadMLPFromFile("../model/mnist.mlp");
	MLP n = create_mlp(784, 250, 10);
	//SGD o = init_sgd(n.params, n.param_grad, n.num_params);
	Momentum o = create_optimizer(Momentum, n);
	o.alpha = 0.001;
	o.beta = 0.99;

	//n.batch_size = 1;
	size_t epochs = 5;
	int epoch = 0;

	// Training data
	{
		ImageSet training_set;
		size_t height, width;
		openImageSet(&training_set, 47040016, trainset_images, trainset_labels); 
		if(training_set.imgBuff == NULL) {
			printf("WARNING: mnist data set not loaded correctly. Check filenames.\n");
			exit(1);
		}
		printf("Training for %lu epochs.\n", epochs);
		float avgCost = 0;
		clock_t start = clock();
		for(size_t i = 0; i < training_set.numImages * epochs; i++) { // Run for the given number of epochs
			//size_t index = i % training_set.numImages;
			size_t index = rand() % training_set.numImages;
			float *x = img2floatArray(&training_set, index, &height, &width); // Image is returned as a float array
			int correctlabel = label(&training_set, index); // Retrieve the label from the image set.
			CREATEONEHOT(y, 10, correctlabel); // Create a float array for cost
			
			mlp_forward(&n, x);
			float c = mlp_cost(&n, y);
			//printf("got cost %f\n", c);
			if(isnan(c)){
				printf("nan!\n");
				exit(1);
			}
			mlp_backward(&n);
			o.step(o);

		 avgCost += c;

			// Save the state of the network at the end of each epoch.
			if(i % training_set.numImages == 0 && i != 0) {
				float secs = (float)(clock() - start) / CLOCKS_PER_SEC;
				printf("Epoch %d finished in %6.5f seconds, cost %f.\n", epoch++, secs, avgCost / i);
				start = clock();
			}
		}
	}

	// Testing data
	{
		printf("Testing:\n");
		ImageSet testing_set;

		openImageSet(&testing_set, 7840016, testset_images, testset_labels);

		float avgCost = 0;
		float avgCorrect = 0;

		for (int i = 0; i < testing_set.numImages; i++) {
			size_t height, width;
			float *img = img2floatArray(&testing_set, i, &height, &width); // Get image as float array

			mlp_forward(&n, img); // Perform feedforward only, no backprop

			avgCorrect += (n.guess== label(&testing_set, i));
		}
		printf("Accuracy on test set: %f\n", avgCorrect / testing_set.numImages);
	}
}
