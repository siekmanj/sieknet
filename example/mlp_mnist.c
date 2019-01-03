/* Jonah Siekmann
 * 7/24/2018
 * In this file are some tests I've done with the network.
 */
#include <MLP.h>
#include <math.h>
#include <mnist.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

char *modelfile = "../model/mnist.mlp";

char *trainset_images = "../data/mnist/train-images-idx3-ubyte";
char *testset_images = "../data/mnist/t10k-images-idx3-ubyte";

char *trainset_labels = "../data/mnist/train-labels-idx1-ubyte";
char *testset_labels = "../data/mnist/t10k-labels-idx1-ubyte";

int main(void) {
  srand(time(NULL));

  //MLP n = loadMLPFromFile("../saves/mnist_784_20_20_10.mlp");
  MLP n = createMLP(784, 20, 20, 10);
  n.plasticity = 0.03; // 0.05 is a good starting learning rate - generally, the
                       // more layers/neurons, the lower your learning rate
                       // should be.

  // Set some interesting activation functions for variety
//  n.input->output_layer->squish = hypertan;
 // n.output->input_layer->squish = leaky_relu;

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
    for(size_t i = 0; i < training_set.numImages * epochs; i++) { // Run for the given number of epochs
      size_t index = i % training_set.numImages;
      float *img = img2floatArray(&training_set, index, &height, &width); // Image is returned as a float array
                                           																// (must have same dims as input
                                        															    // layer)
      setInputs(&n, img); // Set the activations of the neurons in the input layer.

      int correctlabel = label(&training_set, index); // Retrieve the label from the image set.

      float c = descend(&n, correctlabel); // Perform feedforward &
                                           // backpropagation, and return cost

      avgCost += c;

      // Save the state of the network at the end of each epoch.
      if(i % training_set.numImages == 0 && i != 0) {
        printf("Epoch %d finished, cost %f.\n", epoch++, avgCost / i);
        saveMLPToFile(&n, modelfile);
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
      float *img = img2floatArray(&testing_set, i, &height,
                                  &width); // Get image as float array
      setInputs(&n, img); // Set the activations in the input layer

      feedforward(&n); // Perform feedforward only, no backprop

      int guess = bestGuess(&n); // Get network's best guess

      avgCorrect += (guess == label(&testing_set, i));
    }
    printf("resulting avg correct: %f\n", avgCorrect / testing_set.numImages);
  }
}
