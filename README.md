# Forward Models Tutorial

## TODO

- [ ] Document notebook
- [ ] Document code

## Forward Models

This repository implements a canonical forward model. In model-based reinforcement learning, the role of the forward model is to stand in for an environment. It is often fitted from data to predict the next state, given the current state and action:

    f(s_t, a_t) => s_{t+1}

Sometimes an entire history of prior states and actions for a particular episode are provided to the model:

    f(s_{0..t}, a_{0..t}) => s_{t+1}

Below are several important features for a strong dynamics model...

## Feature Normalization

Always standardize the input features by subtracting the mean and dividing by the standard deviation. Compute the mean and standard deviation from the training data.

## Angles

Always encode angles and other wrapping values as the cosine and sine of radians. This helps the loss function by not overly penalizing e.g. 0 degrees versus 359 degrees.

## State-Action History

Forward models of dynamical systems typically benefit from incorporating a time-based history of the features.

## Target Delta

Always predict the delta from the current state, not the next state directly. This minimizes error by helping the model avoid incorrectly predicting the obvious.

    f(s_t, a_t) => s_{t+1} - s_t

## Target Normalization

Always standardize the targets (deltas) using training data statistics.

## Logit Activation

Omit the logit activation to leave the predictions unbounded. Often the "true" bounds are not known so rescaling to -1..1 and using a tanh activation is not as robust.

## Instantaneous Evaluation

The instantaneous evaluation is the simplest. For each step, predict the next state given a _ground truth_ state and action. Typically we only use this for spot-checking the predictions.

## Rollout Evaluation

The rollout evaluation is the most important because it mimics the usage of the forward model as an environment for an agent. For the first step, predict the next state given a ground truth state and action. For all subsequent steps, predict the next state given the previously predicted state and a ground truth action. This evaluation stresses the temporal generalization of the model. A good rollout is accurate for some number of steps before diverging from the ground truth states.
