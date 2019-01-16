# sieknet
This is a dependency-free neural network/deep learning library I am attempting to write from scratch in C. 

I have used the following resources extensively in the course of this project:

  * Michael Nielsen's excellent [online book](http://www.neuralnetworksanddeeplearning.com)
  * 3blue1brown's [YouTube series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  * Wikipedia's article on [Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) 
  * Andrej Karpathy's [article on RNN's](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  * Eli Bendersky's article on the [softmax function](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
  * Aidan Gomez's blog post on [backpropagating an LSTM cell](https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/)
   
So far, I have implemented:
  * A feed-forward neural network which you can find under `src` directory. 
  * A recurrent neural network, which you can find in the `src` directory. 
  * A long short-term memory (LSTM) network, which you can find in the `src` directory.
  * Backpropagation and backpropagation through time (BPTT).
    
In the coming months, I intend to implement:
  * A GRU (Gated Recurrent Unit)
  * Attention (transformer neural network)
  * Filip Pieknewski's [Predictive Vision Model](https://blog.piekniewski.info/2016/11/04/predictive-vision-in-a-nutshell/)
  
Various demonstrations of how to use the networks can be found in `/example/`, along with a makefile for compiling them.

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
