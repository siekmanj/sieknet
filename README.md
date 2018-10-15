# sieknet
This is a dependency-free neural network/deep learning library I am attempting to write from scratch in C. 

I have used the following resources extensively in the course of this project:

  * Michael Nielsen's excellent [online book](http://www.neuralnetworksanddeeplearning.com)
  * 3blue1brown's [YouTube series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  * Wikipedia's article on [Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) 
  * Andrej Karpathy's [article on RNN's](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  * Eli Bendersky's article on the [softmax function](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
   
So far, I have implemented:
  * A feed-forward neural network which you can find under `/networks/MLP/`. 
  * A recurrent neural network, which you can find under `/networks/RNN/`. 
  * Backpropagation.
  * Softmax activation and cross-entropy cost.
    
In the coming months, I intend to implement:
  * A LSTM network.
  * A convolutional neural network.
    * Geoffrey Hinton's [Capsule network](https://en.wikipedia.org/wiki/Capsule_neural_network)
  * Filip Pieknewski's [Predictive Vision Model](https://blog.piekniewski.info/2016/11/04/predictive-vision-in-a-nutshell/)
  
Various demonstrations of how to use the networks can be found in `/examples/`. Currently, I am  using Shakespeare's sonnets to train a 512x512x512 recurrent network to compose a sonnet. 

To do a quick-and-dirty demo:

```
git clone https://github.com/siekmanj/sieknet
cd sieknet/build
make recurrent_demo && ./recurrent_demo.out
```
Below is a fairly typical sample you may get from running the demo. As you can see, the network is able to learn a variety of words, but likes to invent its own, just like Shakespeare ;)
```
  Cate' thou from my graving,
  And so the raster fat hath my sine door heat a too lease lat the ware and self, ede boved and heat,
  Cave all the bat that is now,
  Mor my merpon thy live, woth live, aga me thy love, with seep love,
  Coughts fall the beartare thee,
  Aha leare the true ia por fore and fear,
  Which thee and from thee how faul cate,
  Chat for my me belf the winthe brackressall,
  That hel fairied neap age byaur fae cans and beauty bear,  
  Cire my love anged soe collare again therrent thou anl me borathe lais, and the waright,
  Ar hadrence thee foace for where beat ore born, are 
  Or hat all a and the pragalan,
  And the the prayay forn with and woll, alater hook ore and heart
```

