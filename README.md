# Forward Models Tutorial

This repository implements a canonical forward model. In model-based reinforcement learning, the role of the forward model is to stand in for an environment so we can train agents against the model rather than sampling from the real environment which can be expensive and time consuming. The forward model is typically fitted from data to predict the next state, given the current state and action:

    f(s_t, a_t) => s_{t+1}

Sometimes an entire history of prior states and actions for a particular episode are provided to the model:

    f(s_{0..t}, a_{0..t}) => s_{t+1}

Below are several important features for a strong forward model...

## Layer Activations

Use ReLU or Swish (`x * tf.sigmoid(x)`) for dense layers except the logits. Swish tends to perform slightly better due to improved gradient flow and avoids dying activations.

## Layer Initialization

Use orthogonal initialization for recurrent layers to reduce exploding gradients. Use He et al initialization (`tf.initializers.variance_scaling`) for dense layers with a scaling factor of 2.0 for ReLU and Swish.

## Training

When using an LSTM its important to sample different starting conditions otherwise the LSTM will learn to expect an empty hidden state only in a few starting conditions.

## Initial Cell States

Fitting the initial cell states as a free variable can improve results. Adding noise to the initial states can help avoid overfitting.

## State-Transitions

Incorporate curriculum learning to transition from observation-dependent transitions to prediction-dependent transitions over increasing time horizons. See [Recurrent Environment Simulators](https://arxiv.org/abs/1704.02254) for more details.

## Feature Normalization

Always standardize the input features by subtracting the mean and dividing by the standard deviation. Compute the mean and standard deviation from the training data.

## Angles

Always encode angles as the cosine and sine of radians. This helps the loss function by not overly penalizing e.g. 0 degrees versus 359 degrees. Note that the Pendulum-v0 environment does this automatically.

## State-Action History

Forward models of dynamical systems typically benefit from incorporating a time-based history of the features.

## Target Delta

Always predict the delta from the current state, not the next state directly. This minimizes error by helping the model avoid incorrectly predicting the obvious. This also stationarizes the targets which is helpful for time series prediction.

    f(s_t, a_t) => s_{t+1} - s_t

## Target Normalization

Always standardize the targets (deltas) using training data statistics. Alternatively, scale outputs to -1..1 and use a tanh activation. This can work better depending on the scale and variance of the delta states. See next:

## Logit Activation

Typically we omit the logit activation to leave the predictions unbounded. Often the "true" bounds are not known so rescaling to -1..1 and using a tanh activation may not be as robust. If there are hard bounds on the states a tanh activation may work better.

## Instantaneous Evaluation

The instantaneous evaluation is the simplest. For each step, predict the next state given a _ground truth_ state and action. Typically we only use this for spot-checking the predictions.

## Rollout Evaluation

The rollout evaluation is the most important because it mimics the usage of the forward model as an environment for an agent. For the first step, predict the next state given a ground truth state and action. For all subsequent steps, predict the next state given the previously predicted state and a ground truth action. This evaluation stresses the temporal generalization of the model. A good rollout is accurate for some number of steps before diverging from the ground truth states.
