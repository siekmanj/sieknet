# sieknet
## A dependency-free recurrent neural network library written in C
This is a recurrent neural network and deep learning library written in C which implements various machine learning algorithms. I have mostly focused on recurrent and memory-based networks while writing this, because these interest me the most. 

It has no mandatory dependencies and is written completely from scratch - all you need to compile and run this code is `gcc` or any C compiler.

As of April 2019, you can run sieknet on your GPU via OpenCL 1.1. If you don't need to use the GPU, you don't need to worry about installing OpenCL - it is an optional dependency. 

If you would like to use the GPU, you need to #define SIEKNET_USE_GPU when compiling. You can put this in include/conf.h, or declare it with the -D flag (check the Makefile for an example).

Features include:
 - [x] basic multiplayer perceptron (MLP)
 - [x] long short-term memory (LSTM)
 - [x] stochastic gradient descent
 - [x] stochastic gradient descent with momentum
 - [x] backpropagation through time (BPTT)
 - [x] GPU support (via OpenCL)
 
Plans for the near future include:
 - [ ] nesterov's accelerated gradient optimizer
 - [ ] adam stochastic optimizer
 - [ ] vanilla recurrent neural network (implemented but broken)
 - [ ] gated recurrent unit (GRU)
 - [ ] policy gradients and various RL algorithms
 
 Plans for the distant future include:
 - [ ] Filip Pieknewski's [predictive vision model](https://blog.piekniewski.info/2016/11/04/predictive-vision-in-a-nutshell/)
 - [ ] Neural turing machine
 - [ ] Transformer

Everything is written so as to be easily modifiable. Parameters are stored in one large array similar to [Genann](https://github.com/codeplea/genann), so as to allow for alternative training methods like a genetic algorithm.

## Usage
### Optimizers
All networks have a member array that stores the parameters of the network (`n.params`), and an array that stores the gradient of the loss function with respect to the parameters (`n.param_grad`). The `n.param_grad` array is calculated and updated by the networks' `xyz_backward` functions. The two arrays correspond to each other - i.e., the ith index of `n.param_grad` is the gradient of the ith index of `n.params`. This makes writing optimizers particularly straightforward and modular. So far, I have implemented two - stochastic gradient descent and sgd + momentum.

You can create an optimizer using the `create_optimizer` macro, which takes as arguments the type of optimizer and the network it is optimizing (passed by value).

```C
SGD o1 = create_optimizer(SGD, neural_network); //create a stochastic gradient descent optimizer.
Momentum o2 = create_optimizer(Momentum, neural_network); //create a momentum optimizer

o1.step(o1); //Perform a parameter update with sgd using n.param_grad.
o2.step(o2); //Perform a parameter update with momentum using n.param_grad.

```

### Multilayer Perceptron
Create a 2-layer mlp with an input dimension of 784 neurons:
```C
MLP n = create_mlp(784, 50, 10);
```
Create a 3-layer network with an input dimension of 784 neurons:
```C
MLP n = create_mlp(784, 35, 25, 10);
```

Run a single forward/backward step, and update the parameters:
```C
MLP n = create_mlp(2, 16, 2); //Create a 2-layer network with hidden layer size 16.
SGD o = create_optimizer(SGD, n); //Create a stochastic gradient descent optimizer object.
o.learning_rate = 0.01; 

float x[2] = {0.5, 0.1}; //network input
float y[2] = {0.0, 1.0}; //output label

mlp_forward(&n, x); //Run forward pass
float cost = mlp_cost(&n, y); //Evaluate cost of network output
mlp_backward(&n); //Run backward pass (calculate n.param_grad)

o.step(o); //Update the parameters of the network using gradients calculated in mlp_backward.

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
n = load_mlp("../model/file.mlp");
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
You can't change the logistic functions used by the lstm hidden layers, and you probably don't want to either (Schmidhuber & Hochreiter knew what they were doing). You can, however, change the logistic function used by the output layer.
```C
LSTM n = create_lstm(10, 20, 10);
n.output_layer.logistic = relu; 
```

Using the forward/backward pass functions:
```C
LSTM n = create_lstm(2, 5, 2);
Momentum o = create_optimizer(Momentum, n); //Create a momentum optimizer
o.alpha = 0.01;
o.beta = 0.99;

n.seq_len = 3; //How long the time sequences in your data are.
n.stateful = 0; //Reset hidden state & cell state every parameter update.
for(int i = 0; i < 6; i++){
	lstm_forward(&n, x); //Evaluated every i
	float c = lstm_cost(&n, y); //Evaluated every i
	lstm_backward(&n); //Because seq_len=3, the backward pass will only be evaluated when i=2 and i=5

	//Strictly speaking, you can run o.step(o) every timestep, as n.param_grad will be zeroed, so no updates will occur.
	//However, it is inefficient to do so, and may interfere with momentum's averaging.
	//Therefore, I recommend that you only run a parameter update when n.t is zero - having been reset by lstm_backward.
	if(!n.t) o.step(o); //Only run optimizer after gradient is calculated and n.t is reset to 0.
}

```
Note that your `n.seq_len` determines when the backward pass is run. In the above example, the gradient calculation (`lstm_backward()`) is evaluated every third timestep.

You will need to decide how long to make your seq_len. If you use a sequence length longer than a few hundred, you may run into the exploding gradient problem. If your sequence data is particularly long, you can use the `n.stateful` flag to stop recurrent inputs and cell states from being zeroed out after a parameter update. If you absolutely need to use a large sequence length, you can change SIEKNET_MAX_UNROLL_LENGTH to pre-allocate more memory.
```C
n.stateful = 1; //recurrent inputs and cell states won't be reset after a parameter update.
n.seq_len = 30; //run parameter update (backpropagation through time) every 30 timesteps
for(int i = 0; /*forever*/; i++){
	lstm_forward(&n, x);
	lstm_cost(&n, y);
	lstm_backward(&n); //Evaluated every 30 timesteps, does NOT reset lstm states.
	if(!n.t) o.step(o); //Only run optimizer after gradient is calculated and n.t is reset to 0.
		
	if(sequence_is_over)
		wipe(&n);
}
```
If you choose to do this, you will need to reset the states at some point yourself using `wipe()`, as shown above.

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
n = load_lstm("../your/file.lstm");
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

## Samples			
Here is a short sample from an lstm trained on shakespeare's complete works:

    PRINCE. My lord, now, then him in the with thee,
      For speak to the with the part of the wide,
      And with the with a straing to the with the wears and the
      sweet the with the with thee truity, and like the mack
      this the with
      two in betwee the with the discoure did the words
      Thou are not the with with that the with the wit
      sating to prison?
    FALSTAFF. You are the with him the with the body
      make in the with the?
      He will you after thou doth my like and the wite;
      You are let the will become poor and like a such of
      the with the with this with him the will be him
      turn the with the with that the well!
    HOSTESS. No lustion to the within their king of endurs
      the with the wise the
      the with his a bount of themself-
      good true thee did the witch good-ly the within to the was a comfuls
      to the within the with eyes and the withins and
      the will appears, as the wide to himself
Here is a short sample from an lstm trained on the bible:
     
    21:22 And then his son, and through the God of the son. 
    22:22 Rend they shall be to be said, When the sead of the brother which is was the thildren of the son of Judah,
    22:20 And they priest the son of Asseemy to Zenah, and they shall be according, and them he said, 
    22:25 And turne, and the son of Ashemes, and the sears, remembered the son of Abraham,
    22:22 In they shall be and the sead in the princes to their son of David be said, 
    32:22 And they said unto the son of Asster the son of of a aspenters, and they that were before the years,
    22:22 And all the redeemed of dead, and tree ten the shewed when they said.

Here is a short sample from an lstm trained on trump tweets:

    The Fake a great the Fake News and Republicans will be and the Fake News Marking and the Fake Newss and the Fake Kenney Border and the Fake Government #MAGA 
    The Foxar to state to the total Putings for World. The Debal Security and the Fake News Mark Canferer #MAGA 
    The Democrats so much to the same to the State of the Fake Media News Market
    The Fake News-- and 4 alanch to running tough and a $500 and over a great the highest to the Fake News Marker Care
