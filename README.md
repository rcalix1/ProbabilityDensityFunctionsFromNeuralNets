## Probability Density Functions (PDFs) from Neural Networks

Probability Density Functions (PDFs) have many applications in the Engineering Sciences, Industrial Engineering, Finance, and many other fields. They are used in regression type models which predict real valued numbers in a given range. The goal of the neural network model is to predict a real valued number but also provide a probability of that number. The PDF should add to 1. 

The general function is as follows:

$ y = g(x) = f(x) + e(x) $

where  f(x) is the function and e(x) is the noise or error. 

It is assumed that a deep neural network will learn the f(x) and e(x) functions together and so g(x) is the neural network. Unlike regular regression models which learn to predict y values given input values x. In PDF shaping, the neural net learns to predict probabilities given x inputs. Here given a range of values, the model you indicate the probability of that range. 

In an industrial process, you can learrn the shape of your real PDF function by comparing it to an ideal distribution function. This is called PDF shaping and it has many applications in stochastic control systems. 

The loss function you are trying to optimize is as follows:

$   J = min \sum \limits _{i} ^{n} (g(x) - I(x))^2 $

where I(x) is the ideal function also know as the impulse function which can be defined as a gaussian function withe very narrow standard deviation and the mean of the target output variable. 

$ \large  I(x) =  \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{(x - \mu)^2}{2 \sigma ^2}   $

The code in this repo (written in python) is the companion to the notebook:

### Notebook

Machine Learning for System Control


### Contact
Ricardo A. Calix, Ph.D.

### Notices
Released as is with no warranty

