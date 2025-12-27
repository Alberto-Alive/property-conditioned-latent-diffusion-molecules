Sometimes to create one must destroy. 

A DDPM model actually learns how to turn a noise structure into a desired one by gradually removing noise.
Don't get too excited as this is similar to what a GPT style Transformer does but instead of predicting the next token in a sequence, it predicts less noisy versions of a structure. This means machine learning is about prediction, patterns are useful for prediction and the one who knows the future controls the present. In fact eveything is about order, then in order you find patterns, and with patterns you can predict the future.

Apparently the key to agi is continual learning yet today's ai is built like a chain so modifing one link breaks the whole chain.

## Symbols dictionary

* **(x_0)**: the *real data sample* (e.g., a clean image).
* **(x_t)**: the same sample after *t steps of noise* (more noisy as (t) grows).
* **(x_{0:T})**: shorthand for the whole sequence (x_0, x_1, \dots, x_T).
* **(x_{1:T})**: the “latent” (hidden) noisy steps (x_1, \dots, x_T).
* **(p_\theta(\cdot))**: the *model’s* probability distribution, with learnable parameters (\theta) (the neural net weights).
* **(q(\cdot))**: the *fixed* “true / chosen” diffusion process (the noising process).
* **(\int \cdots , dx_{1:T})**: “sum over / average over all possible values of (x_1 \dots x_T)” (continuous version of summing).
* **Markov chain**: “each step depends only on the previous step,” e.g. (x_t) depends only on (x_{t-1}).
* **(\mathcal N(x; \mu, \Sigma))**: a Gaussian (normal) distribution over (x), with mean (\mu) and covariance (\Sigma).
* **(I)**: identity matrix (think “unit variance, no correlations”).
* **(\beta_t)**: how much noise you add at step (t) (noise schedule).
* **(\alpha_t := 1-\beta_t)**: how much “signal” you keep at step (t).
* **(\bar\alpha_t := \prod_{s=1}^t \alpha_s)**: how much original signal remains after *t* steps total.
* **(\mu_\theta(x_t,t))** and **(\Sigma_\theta(x_t,t))**: the model’s predicted mean and variance for the reverse step.
