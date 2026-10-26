# Questions:

1. What would change if the network had 1 layer before the final classification layer?

2. What would change if the network had 3 layers before the final classification layer?

3. What if the hidden layers had 8 neurons per layer?

4. What if the hidden layers had 32 neurons per layer?

5. What if the hidden layers had 64 neurons per layer?

6. What would happen if the loss function were changed to 'mse'?

7. What would happen if the 'tanh' function were used instead of 'relu'?

8. How does batch size affect training?

The smaller batch size, the more often the model updates its weights. As a result, it reaches optimal weights in fewer epochs, but it also starts overfitting sooner. However, each epoch takes longer to process.

If the batch size is large, the weights are updated less often, but each update is more precise.

When the batch size is small, the gradient is noisy and imprecise, but this noise can help the model escape suboptimal local minima.
