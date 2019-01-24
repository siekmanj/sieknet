# sieknet
## A dependency-free neural network library written in C
This is a neural network and deep learning library written in C which implements various machine learning algorithms. It has no dependencies and is written completely from scratch - all you need to compile and run this code is `gcc` or any C compiler.

Features include:
 - [x] basic multiplayer perceptron (MLP)
 - [x] long short-term memory (LSTM)
 - [x] backpropagation (gradient descent)
 - [x] backpropagation through time (BPTT)
 
Plans for the near future include:
 - [ ] Adam stochastic optimizer
 - [ ] Momentum stochastic optimizer
 - [ ] vanilla recurrent neural network (implemented but broken)
 - [ ] gated recurrent unit (GRU)
 - [ ] policy gradients and various RL algorithms
 
 Plans for the distant future include:
 - [ ] Filip Pieknewski's [predictive vision model](https://blog.piekniewski.info/2016/11/04/predictive-vision-in-a-nutshell/)
 - [ ] Neural turing machine
 - [ ] Transformer

Everything is written so as to be easily modifiable. Parameters are stored in one large array similar to [Genann](https://github.com/codeplea/genann), so as to allow for alternative training methods like a genetic algorithm.

## Usage
#### Multilayer Perceptron
Create a 2-layer mlp with an input dimension of 784 neurons:
```C
MLP n = create_mlp(784, 50, 10);
```
Create a 3-layer network with an input dimension of 784 neurons:
```C
MLP n = create_mlp(784, 35, 25, 10);
```

Run a single forward/backward step:
```C
MLP n = create_mlp(2, 16, 2);
n.learning_rate = 0.01; //Set the learning rate

float x[2] = {0.5, 0.1}; //network input
float y[2] = {0.0, 1.0}; //output label

mlp_forward(&n, x); //Run forward pass
float cost = mlp_cost(&n, y); //Evaluate cost of network output
mlp_backward(&n); //Run backward pass (update parameters)

dealloc_mlp(&n); //Free the network's memory from the heap
```

You can also just run the forward pass:
```C
while(1){
    mlp_forward(&n, x);
    printf("network output: %d\n", n.guess);
}
```

By default, hidden layers will use the sigmoid logistic function and the output layer will use softmax. However, you can use any of the other activation functions implemented:
```C
MLP n = create_mlp(10, 50, 20, 35, 5);
n.layers[0].logistic = hypertan; //The layer of size 50 will now use tanh activation
n.layers[1].logistic = softmax; //The layer of size 20 will now use softmax activation
n.layers[2].logistic = relu; //The layer of size 35 will now use rectified linear unit activation
n.layers[3].logistic = sigmoid; //The output layer of size 5 will now use sigmoid activation
```

Save/load models to disk:
```C
save_mlp(&n, "../model/file.mlp");
dealloc_mlp(&n);
MLP b = load_mlp("../model/file.mlp");
```
#### Long Short-Term Memory
You can create an LSTM just like an MLP. The first number provided will be the input dimension to the network, subsequent numbers will be the size of hidden layers, and the final number will be the size of the softmax output layer.
```C
LSTM n = create_lstm(10, 25, 10); //Creates an lstm with an input dim of 10, hidden size of 25, softmax layer of 10.
```
You can have as many hidden layers as you'd like, just as with an MLP.
```C
LSTM n = create_lstm(10, 50, 25, 3, 350, 10); //Ditto, but with four hidden lstm layers of size 50, 25, 3, and 350.
```
You can't change the logistic functions used by the lstm hidden layers, and you probably don't want to either (Hochreiter knew what he was doing). You can, however, change the logistic function used by the output layer, since it's just an MLP.
```C
LSTM n = create_lstm(10, 20, 10);
n.output_layer.layers[0].logistic = relu; 
```
The syntax looks a little bit gross, however. This is because the LSTM implementation uses an MLP network as its output layer. I might at some point change the implementation to use only a single MLP layer - but this would require rewriting significant chunks of the cost function code.

Using the forward/backward pass functions:
```C
LSTM n = create_lstm(2, 5, 2);
n.learning_rate = 0.01; //The learning rate the network will use
n.seq_len = 3; //How long the time sequences in your data are.
n.stateful = 0; //Reset hidden state & cell state every parameter update.
for(int i = 0; i < 6; i++){
    lstm_forward(&n, x); //Evaluated every i
    float c = lstm_cost(&n, y); //Evaluated every i
    lstm_backward(&n); //Because seq_len=3, the backward pass will only be evaluated when i=2 and i=5
}

```
Note that your `n.seq_len` determines when the backward pass is run. In the above example, the parameter update (`lstm_backward()`) is evaluated every third timestep.

You will need to decide how long to make your seq_len, but I recommend somewhere between 10 and 35. If you use a sequence length longer than 35, you may run into the exploding gradient problem. If your sequence data is longer than 35 timesteps, you can use the `n.stateful` flag to stop recurrent inputs and cell states from being zeroed out after a parameter update.
```C
n.stateful = 1; //recurrent inputs and cell states won't be reset after a parameter update.
n.seq_len = 30; //run parameter update (backpropagation through time) every 30 timesteps
for(int i = 0; /*forever*/; i++){
    lstm_forward(&n, x);
    lstm_cost(&n, y);
    lstm_backward(&n); //Evaluated every 30 timesteps, does NOT reset lstm states.
    
    if(sequence_is_over)
      wipe(&n);
}
```
If you choose to do this, you will need to reset the states at some point yourself using `wipe()`, as shown above.

If you absolutely need to train over sequences longer than 35 timesteps, the library supports up to 200 timesteps. There is a `#define MAX_UNROLL_LENGTH 200` in `/include/lstm.h` which you can modify if you need more than that. I don't recommend doing this, as backpropagation through time is a fairly memory-intensive algorithm.

If you just want to run the network without training, you can do so like this:
```C
for(int i = 0; /*forever*/; i++){
    lstm_forward(&n, x);
    printf("network output: %d\n", n.guess);
}
```
Saving and loading the network is straightforward:
```C
save_lstm(&n, "../your/file.lstm");
dealloc_lstm(&n);
LSTM n = load_lstm("../your/file.lstm");
```
Various demonstrations of how to use the library can be found in `/example/`, along with a makefile for compiling them.

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
