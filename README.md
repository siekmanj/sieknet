# sieknet
## A dependency-free neural network library written in C
This is a neural network and deep learning library written in C which implements various machine learning algorithms. It has no dependencies - all you need to compile and run this code is `gcc` or any C compiler.

Features include:
 - [x] basic multiplayer perceptron (MLP)
 - [x] long short-term memory (LSTM)
 - [x] backpropagation (gradient descent)
 - [x] backpropagation through time (BPTT)
 
Plans for the near future include:
 - [ ] vanilla recurrent neural network (implemented but broken)
 - [ ] gated recurrent unit (GRU)
 - [ ] policy gradients and various RL algorithms
 - [ ] support for batch sizes greater than 1 (currently all updating is done online).
 
 Plans for the distant future include:
 - [ ] Filip Pieknewski's [Predictive Vision Model](https://blog.piekniewski.info/2016/11/04/predictive-vision-in-a-nutshell/)
 - [ ] neural turing machine

Everything is written so as to be easily modifiable. Parameters are stored in one large array similar to [Genann](https://github.com/codeplea/genann), so as to allow for alternative training methods like a genetic algorithm.

## Usage
Create a 2-layer mlp with an input dimension of 784 neurons:
```C
MLP n = createMLP(784, 50, 10);
```
Create a 3-layer network with an input dimension of 784 neurons:
```C
MLP n = createMLP(784, 35, 25, 10);
```

Run a single forward/backward step:
```C
MLP n = createMLP(2, 16, 2);
n.learning_rate = 0.01; //Set the learning rate

float x[2] = {0.5, 0.1}; //network input
float y[2] = {0.0, 1.0}; //output label

mlp_forward(&n, x); //Run forward pass
float cost = n.cost(&n, y); //Evaluate cost of network output
mlp_backward(&n); //Run backward pass (update parameters)

dealloc_network(&n); //Free the network's memory from the heap
```

By default, hidden layers will use the sigmoid logistic function and the output layer will use softmax. However, you can use any of the other activation functions implemented:
```C
MLP n = createMLP(10, 50, 20, 35, 5);
n.layers[0].logistic = hypertan; //The layer of size 50 will now use tanh activation
n.layers[1].logistic = softmax; //The layer of size 20 will now use softmax activation
n.layers[2].logistic = relu; //The layer of size 35 will now use rectified linear unit activation
n.layers[3].logistic = sigmoid; //The output layer of size 5 will now use sigmoid activation
```

Save/load models to disk:
```C
save_mlp(&n, "../model/file.mlp");

MLP b = load_mlp("../model/file.mlp");
```
The LSTM implementation is in the process of being rewritten, and so I do not recommend using it. However, if you would like to, the interface is very similar to the MLP interface.

Various demonstrations of how to use the networks can be found in `/example/`, along with a makefile for compiling them.

## References
I have used the following resources extensively in the course of this project:

  * Michael Nielsen's excellent [online book](http://www.neuralnetworksanddeeplearning.com)
  * 3blue1brown's [YouTube series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  * Wikipedia's article on [Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) 
  * Andrej Karpathy's [article on RNN's](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  * Eli Bendersky's article on the [softmax function](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
  * Aidan Gomez's blog post on [backpropagating an LSTM cell](https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/)

      
Here is a short sample from an lstm trained for 3 epochs on shakespeare's complete works:

    IMOGEN. I'll be stay songer for beardess
      And stranger be some before that be
      If is the servents and bearded
      As books bearthers. I'll be for hath before beard
      And stronger that be staing.
      As I have be the forthers,
      And streath of my bearded be for
      And streather.
      As I see that be some before beard
      As forthing be some beforest beard
      As forthing bearst for his beath
      As be bounders forthers,
      As be againg to be stords
      And streather. I'll be some be for here.
    Enter GERIAN. I'll be as all be forthing bears
      And stranger.
      As I stall be best be forthers,
      And stranger be to stronger.
      As I seed be as all be the forthy sours
