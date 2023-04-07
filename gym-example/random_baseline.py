import gym
import gym_example

def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(env.max_episode_steps):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    return sum_reward

env = gym.make("henv")

#sum_reward = run_one_episode(env)
#print(sum_reward)


history = []
for _ in range(100):
    sum_reward = run_one_episode(env)
    history.append(sum_reward)
avg_sum_reward = sum(history) / len(history)
print("\nbaseline cumulative reward: ", round(avg_sum_reward, 2))