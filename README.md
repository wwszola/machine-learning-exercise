Implementation exercises for various machine learning topics.
Based on [CS229 Lecture Notes](https://cs229.stanford.edu/main_notes.pdf)

## Notes
### Linear regression
Both batch and stochastic gradient descent update rule seems to be very sensitive to learning rate value.
#### Hypothesis: To be successful, an algorithm should implement:
- convergence condition instead of max iteration condition
- predictor variables should be normalized