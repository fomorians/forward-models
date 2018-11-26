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
    learning_rate = attr.ib(default=5e-2)
    batch_size = attr.ib(default=1024)
    clip_grads = attr.ib(default=1.0)
    epochs = attr.ib(default=100)
    max_steps = attr.ib(default=200)
    episodes_train = attr.ib(default=1000)
    episodes_eval = attr.ib(default=10)


def create_dataset(tensors,
                   batch_size=None,
                   shuffle=False,
                   shuffle_buffer_size=10000):
    with tf.device('/cpu:0'):
        if batch_size is not None:
            dataset = tf.data.Dataset.from_tensor_slices(tensors)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            dataset = dataset.batch(batch_size=batch_size)
            dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_tensors(tensors)
        return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    print(args)

    params = Params()
    print(params)

    tf.enable_eager_execution()

    env = PendulumEnv()

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    rollout = Rollout(env, max_steps=params.max_steps)

    (states_train, actions_train, rewards_train, next_states_train,
     weights_train) = rollout(
         lambda state: env.action_space.sample(),
         episodes=params.episodes_train)
    (states_eval, actions_eval, rewards_eval, next_states_eval,
     weights_eval) = rollout(
         lambda state: env.action_space.sample(),
         episodes=params.episodes_eval)

    deltas_train = next_states_train - states_train
    deltas_eval = next_states_eval - states_eval

    dataset_train = create_dataset(
        (states_train, actions_train, deltas_train, weights_train),
        batch_size=params.batch_size,
        shuffle=True)
    dataset_eval = create_dataset(
        (states_eval, actions_eval, deltas_eval, weights_eval),
        batch_size=params.batch_size,
        shuffle=True)

    state_normalizer = Normalizer(
        loc=states_train.mean(axis=(0, 1)),
        scale=states_train.std(axis=(0, 1)))
    delta_normalizer = Normalizer(
        loc=deltas_train.mean(axis=(0, 1)),
        scale=deltas_train.std(axis=(0, 1)))
    action_normalizer = Normalizer(
        loc=actions_train.mean(axis=(0, 1)),
        scale=actions_train.std(axis=(0, 1)))

    model = ForwardModel(output_units=env.observation_space.shape[-1])
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    global_step = tf.train.create_global_step()

    summary_writer = tf.contrib.summary.create_file_writer(
        logdir=args.job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    checkpoint = tf.train.Checkpoint(
        state_normalizer=state_normalizer,
        delta_normalizer=delta_normalizer,
        action_normalizer=action_normalizer,
        model=model,
        optimizer=optimizer,
        global_step=global_step)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path)

    for epoch in range(params.epochs):
        for states, actions, deltas, weights in dataset_train:
            states_norm = state_normalizer(states)
            deltas_norm = delta_normalizer(deltas)
            actions_norm = action_normalizer(actions)

            with tf.GradientTape() as tape:
                deltas_norm_pred = model(
                    states_norm, actions_norm, training=True, reset_state=True)

                loss = tf.losses.mean_squared_error(
                    predictions=deltas_norm_pred,
                    labels=deltas_norm,
                    weights=weights)

            grads = tape.gradient(loss, model.trainable_variables)
            grads, grad_norm = tf.clip_by_global_norm(grads, params.clip_grads)
            grads_and_vars = zip(grads, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            grad_norm_clip = tf.global_norm(grads)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/train', loss)
                tf.contrib.summary.scalar('grad_norm', grad_norm)
                tf.contrib.summary.scalar('grad_norm/clip', grad_norm_clip)

        for states, actions, deltas, weights in dataset_eval:
            states_norm = state_normalizer(states)
            deltas_norm = delta_normalizer(deltas)
            actions_norm = action_normalizer(actions)

            deltas_norm_pred = model(
                states_norm, actions_norm, training=False, reset_state=True)

            loss = tf.losses.mean_squared_error(
                predictions=deltas_norm_pred,
                labels=deltas_norm,
                weights=weights)

            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/eval', loss)

    checkpoint_path = os.path.join(args.job_dir, 'ckpt')
    checkpoint.save(checkpoint_path)


if __name__ == '__main__':
    main()
