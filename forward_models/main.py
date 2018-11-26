import os
import attr
import random
import argparse
import numpy as np
import tensorflow as tf

from gym.envs.classic_control import PendulumEnv

from forward_models.model import Normalizer, ForwardModel
from forward_models.rollout import Rollout


@attr.s
class Params:
    """
    Define hyperparameters for the model and training procedure.

    These could also be constants at the top of the file but having a class
    makes persistence eaier.

    Args:
        learning_rate: Properly tuned learning rate.
        batch_size: Batch size to optimize GPU utilization performance.
        grad_clipping: Gradient clipping to prevent exploding gradients.
        epochs: Number of training epochs to ensure convergence.
                Small errors compound so be sure this is sufficient.
        max_episode_steps: Maximum length of an episode.
        episodes_train: Number of training episodes to sample.
        episodes_eval: Number of evaluation episodes to sample.
    """

    learning_rate = attr.ib(default=1e-2)
    batch_size = attr.ib(default=1024)
    grad_clipping = attr.ib(default=1.0)
    epochs = attr.ib(default=200)
    max_episode_steps = attr.ib(default=200)
    episodes_train = attr.ib(default=1000)
    episodes_eval = attr.ib(default=10)


def create_dataset(tensors,
                   batch_size=None,
                   shuffle=False,
                   shuffle_buffer_size=10000):
    """
    This helper function creates a TensorFlow dataset with batching and
    shuffling.

    Args:
        tensors: A tuple of tensors to create a dataset from.
        batch_size: An integer of the desired batch size.
        shuffle: A boolean to enable/disable shuffling of the dataset.
        shuffle_buffer_size: An integer of how many samples are shuffled.

    Returns:

        A tf.data.Dataset.
    """

    # always place the dataset on the CPU
    with tf.device('/cpu:0'):
        if batch_size is not None:
            dataset = tf.data.Dataset.from_tensor_slices(tensors)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            dataset = dataset.batch(batch_size=batch_size)
            # improve performance by pre-fetching batches
            dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensors(tensors)
        return dataset


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    print(args)

    # create the hyperparameters
    params = Params()
    print(params)

    # enable TF Eager
    tf.enable_eager_execution()

    # create the environment
    env = PendulumEnv()

    # set random seeds for reproducibility and
    # easier comparisons between experiments
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create a rollout class, used to sample data from the environment
    rollout = Rollout(env, max_episode_steps=params.max_episode_steps)

    # sample training and evaluation rollouts from the environment
    # using a random policy
    (states_train, actions_train, rewards_train, next_states_train,
     weights_train) = rollout(
         lambda state: env.action_space.sample(),
         episodes=params.episodes_train)
    (states_eval, actions_eval, rewards_eval, next_states_eval,
     weights_eval) = rollout(
         lambda state: env.action_space.sample(),
         episodes=params.episodes_eval)

    # compute deltas between the next state and the current state
    # to use as targets
    deltas_train = next_states_train - states_train
    deltas_eval = next_states_eval - states_eval

    # create datasets for training and evaluation
    dataset_train = create_dataset(
        (states_train, actions_train, deltas_train, weights_train),
        batch_size=params.batch_size,
        shuffle=True)
    dataset_eval = create_dataset(
        (states_eval, actions_eval, deltas_eval, weights_eval),
        batch_size=params.batch_size,
        shuffle=True)

    # create normalizers for the features and targets
    state_normalizer = Normalizer(
        loc=states_train.mean(axis=(0, 1)),
        scale=states_train.std(axis=(0, 1)))
    delta_normalizer = Normalizer(
        loc=deltas_train.mean(axis=(0, 1)),
        scale=deltas_train.std(axis=(0, 1)))
    action_normalizer = Normalizer(
        loc=actions_train.mean(axis=(0, 1)),
        scale=actions_train.std(axis=(0, 1)))

    # create a forward model
    model = ForwardModel(output_units=env.observation_space.shape[-1])

    # create an Adam optimizer which is slightly easier to tune than momentum
    # momentum typically provides better results when properly tuned
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)

    # create global step
    global_step = tf.train.create_global_step()

    # create a checkpoint with all objects with variables so it can be restored
    checkpoint = tf.train.Checkpoint(
        state_normalizer=state_normalizer,
        delta_normalizer=delta_normalizer,
        action_normalizer=action_normalizer,
        model=model,
        optimizer=optimizer,
        global_step=global_step)

    # restore a checkpoint if it exists
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    # create a summary writer for TensorBoard
    summary_writer = tf.contrib.summary.create_file_writer(
        logdir=args.job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    # iterate for some number of epochs over the datasets
    for epoch in range(params.epochs):

        # loop over the training dataset
        for states, actions, deltas, weights in dataset_train:
            # normalize features and targets
            states_norm = state_normalizer(states)
            deltas_norm = delta_normalizer(deltas)
            actions_norm = action_normalizer(actions)

            # compute a forward pass and loss inside with a gradient tape so
            # the trainble variables are watched for gradient computation
            with tf.GradientTape() as tape:
                # compute a forward pass ensuring the RNN state is reset
                deltas_norm_pred = model(
                    states_norm, actions_norm, training=True, reset_state=True)

                # compute the training loss
                # - use mean squared error for most regression problems
                # - optionally: use a Huber loss if there are lots of outliers
                # due to noise that cannot be filtered for some reason
                # - be sure to weight the loss so empty steps are not included
                loss = tf.losses.mean_squared_error(
                    predictions=deltas_norm_pred,
                    labels=deltas_norm,
                    weights=weights)

            # compute gradients
            grads = tape.gradient(loss, model.trainable_variables)

            # clip the gradients by their global norm
            # returns gradients and global norm before clipping
            grads, grad_norm = tf.clip_by_global_norm(grads,
                                                      params.grad_clipping)

            # update the model
            grads_and_vars = zip(grads, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # compute the clipped gradient norm for summaries
            grad_norm_clip = tf.global_norm(grads)

            # log training summaries, including clipped and unclipped grad norm
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/train', loss)
                tf.contrib.summary.scalar('grad_norm', grad_norm)
                tf.contrib.summary.scalar('grad_norm/clip', grad_norm_clip)

        # loop over the evaluation dataset
        for states, actions, deltas, weights in dataset_eval:
            # normalize features and targets
            states_norm = state_normalizer(states)
            deltas_norm = delta_normalizer(deltas)
            actions_norm = action_normalizer(actions)

            # compute a forward pass ensuring the RNN state is reset
            deltas_norm_pred = model(
                states_norm, actions_norm, training=False, reset_state=True)

            # compute the evaluation loss
            loss = tf.losses.mean_squared_error(
                predictions=deltas_norm_pred,
                labels=deltas_norm,
                weights=weights)

            # log evaluation summaries
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/eval', loss)

    # save a checkpoint after training
    checkpoint_path = os.path.join(args.job_dir, 'ckpt')
    checkpoint.save(checkpoint_path)


if __name__ == '__main__':
    main()
