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


int main(int argc, char *argv[]){
  srand(time(NULL));
	if(argc < 2 || strlen(argv[1]) == 0){
		printf("You must provide a valid filename.\n");
		return -1;
	}
	MLP n = initMLP();
	addLayer(&n, 28*28); //input layer
	addLayer(&n, 15);
	addLayer(&n, 10); //output layer

	size_t epochs = 30;
	int epoch = 0;

	//Training data
	{
		ImageSet training_set;
		size_t height, width;
		openImageSet(&training_set, 47040016, "../mnist/train-images-idx3-ubyte", "../mnist/train-labels-idx1-ubyte");

		float avgCost = 0;

		for(size_t i = 0; i < training_set.numImages * epochs; i++){
			size_t index = i % training_set.numImages;
			float* img = img2floatArray(&training_set, index, &height, &width);
      int correctlabel = label(&training_set, index);

			setInputs(&n, img);
			float c = descend(&n, correctlabel);
      if(isnan(c)){
        printf("cost was nan: %f\n", c);
        while(1);
      }
			avgCost += c;

      if(rand() % 5000 == 0){
        printf("label: %d, cost: %f\n", correctlabel, c);
        printOutputs(n.output);
      }

			if(i % training_set.numImages == 0 && i != 0){
				printf("Epoch %d finished, cost %f.\n", epoch++, avgCost/i);
				char buff[100];
				memset(buff, '\0', 100);
				strcat(buff, "../saves/");
				strcat(buff, argv[1]);
				strcat(buff, ".mlp");
        saveMLPToFile(&n, buff);
			}
		}
		printf("\nAvg training cost: %f / %u * %d = %f\n", avgCost, training_set.numImages, epoch, avgCost / (training_set.numImages*epochs));
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
			float* img = img2floatArray(&testing_set, i, &height, &width);
			setInputs(&n, img);
			feedforward(&n);
			int guess = bestGuess(&n);
			avgCorrect += guess == label(&testing_set, i);
		}
		printf("resulting avg correct: %f\n", avgCorrect/testing_set.numImages);
	}
}
