import random
from time import sleep
import gym
import numpy as np
import os


def clear():
    return os.system('clear')


def initialise_world():
    env = gym.make("Taxi-v3").env
    return env, np.zeros([env.observation_space.n, env.action_space.n])


def print_frames(frames, episode_number):
    for i, frame in enumerate(frames):
        clear()
        print(frame['frame'])
        print(f"Episode: {episode_number}")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


def train_model(env, q_table):
    episode_results = []
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    for i in range(1001):
        state = env.reset()
        frames = []
        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * \
                (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            }
            )

            state = next_state
            epochs += 1

        if i % 100 == 0 or i == 0:
            episode_results.append({'epochs': epochs,
                                    'frames': frames, 'penalties': penalties,
                                    'episode_number': i})
            print(f"Episode {i}")
    return episode_results


if __name__ == "__main__":
    env, q_table = initialise_world()
    results = train_model(env, q_table)
    for episode in results:
        print_frames(episode.get("frames"), episode.get("episode_number"))
