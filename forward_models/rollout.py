import numpy as np


class Rollout:
    """
    Define a rollout given an environment and maximum number of rollout steps.

    Call the class to accumulate rollout data from the environment.

    Args:
        env: Environment to use for rollouts.
        max_episode_steps: Maximum length of an episode.
    """

    def __init__(self, env, max_episode_steps):
        self.env = env
        self.max_episode_steps = max_episode_steps

    def __call__(self, policy, episodes):
        """
        Generate transitions from the environment.

        Args:
            policy: A function which returns an action given a state.
            episodes: Number of episodes to sample.

        Returns:
            Transitions from the rollouts:

            states: States.
                (shape [episodes, max_episode_steps, observation_size])
            actions: Actions.
                (shape: [episodes, max_episode_steps, action_size])
            rewards: Rewards.
                (shape: [episodes, max_episode_steps])
            next_states: Next states.
                (shape: [episodes, max_episode_steps, observation_size])
            weights: Indicator of valid states. Used to mask losses.
                (shape: [episodes, max_episode_steps])
        """

        observation_size = self.env.observation_space.shape[-1]
        action_size = self.env.action_space.shape[-1]

        states = np.zeros(
            shape=(episodes, self.max_episode_steps, observation_size),
            dtype=np.float32)
        actions = np.zeros(
            shape=(episodes, self.max_episode_steps, action_size),
            dtype=np.float32)
        rewards = np.zeros(
            shape=(episodes, self.max_episode_steps), dtype=np.float32)
        next_states = np.zeros(
            shape=(episodes, self.max_episode_steps, observation_size),
            dtype=np.float32)
        weights = np.zeros(
            shape=(episodes, self.max_episode_steps, observation_size),
            dtype=np.float32)

        for episode in range(episodes):
            # reset the environment each episode
            state = self.env.reset()

            for step in range(self.max_episode_steps):
                # sample an action from the policy
                action = policy(state)

                # step the environment
                next_state, reward, done, _ = self.env.step(action)

                # store transitions
                states[episode, step] = state
                actions[episode, step] = action
                rewards[episode, step] = reward
                next_states[episode, step] = next_state
                weights[episode, step] = 1.0

                # break when episode terminates
                if done:
                    break

                state = next_state

        return states, actions, rewards, next_states, weights
