## Probability Density Functions (PDFs) from Neural Networks

Modeling Probability Density Functions (PDFs) from neural nets has many applications in the Engineering Sciences, Industrial Engineering, Finance, and many other fields. These PDF predicting models are used in regression type problems which predict real valued numbers in a given range. The goal of the neural network model is to learn how to represent the probability distribution of the underlying data. The PDF should add to 1. 

Two important types are: 

* Mixture Density Networks (MDN)
* PDF Shaping

##  Mixture Density Networks (MDN)

For a detail description of MDNs see my Jupyter notebook here: https://github.com/rcalix1/ProbabilityDensityFunctionsFromNeuralNets/blob/main/MixtureDensityNetworks.ipynb

In essence, MDNs for the gaussian case are neural nets that given x inputs, can learn to predict a y value plus its max variance range. The neural net has 2 outputs instead of just one. The predicted y outputs can be thought of as the mean and sigma values given input x. 

## PDF Shaping

In an industrial process, you can learn the shape of your real PDF function by comparing it to an ideal distribution function. This is called PDF shaping and it has many applications in stochastic control systems. Literature on PDF shaping can be found here (refs).

The general function of some random variable in a process can be represented as follows:

$ y = g(x) = f(x) + e(x) $

where  f(x) is the function and e(x) is the noise or error. 

It is assumed that a deep neural network will learn the f(x) and e(x) functions together and so g(x) will be learned using a neural network. Unlike regular regression models which learn to predict y values given input values x. In PDF shaping, the neural net learns to predict probabilities given x inputs. Here, given a range of values, the model should predict the probability of that range. 

The PDF shape of g(x) is not known and it should not be assumed to have a normal distribution. Since the true distribution is not known, years of research (refs) have shown that a next best approach is to approximate this unknown distribution to an ideal distribution. Therefore, the goal in PDF shaping is to force the true distribution to become as much as possible like the normal distribution. 

As such, the loss function you are trying to optimize is as follows:

$   J = min \sum \limits _{i} ^{n} (g(x) - I(x))^2 $

where I(x) is the ideal function also know as the impulse function which can be defined as a gaussian function withe very narrow standard deviation and the mean of the target output variable. 

$ \large  I(x) =  \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{(x - \mu)^2}{2 \sigma ^2} }  $

The code in this repo (written in python) is the companion to the notebook:

### Notebook

Machine Learning for System Control


### Contact
Ricardo A. Calix, Ph.D.

### Notices
Released as is with no warranty

