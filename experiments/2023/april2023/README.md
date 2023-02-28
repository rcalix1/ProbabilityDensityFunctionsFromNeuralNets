## Experiment 1 using the Hong dataset

Several proofs showing how PDF shaping works.

## Training with linear plus non linear

Using the linear plus non-linear NN architecture, the model achieves:

* for training

Training loss: tensor(0.0654, grad_fn=<MseLossBackward0>)

Training R**2: 0.928283663941792

* for testing

Test loss - scaled: tensor(0.0764, grad_fn=<MseLossBackward0>)

Testing R**2 - scaled: 0.9152176200649317

Testing R**2 - Output: 0 o_y 0.915217627422392

Using PDF shaping does not significantly improve results. The error distribution after linear plus non linear training is as follows

