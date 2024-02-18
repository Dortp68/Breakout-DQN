from model import *
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import time
import json

matplotlib.use('qt5Agg')

def check_env():
    observation = env.reset()
    done = False
    plt.ion()
    while not done:
        plt.clf()
        obs, _, done, _, inf = env.step(env.action_space.sample())
        plt.title("Agent observation (4 frames left to right)")
        plt.imshow(obs.transpose([1, 0, 2]).reshape([env.observation_space.shape[-1], -1]))
        plt.draw()
        plt.gcf().canvas.flush_events()
    env.close()
    plt.ioff()
    plt.show()


def train(episodes=40000, print_every=250, save_every=1000):
    dqn = DQN()
    all_rewards = []
    state_rewards = []
    num_of_succeeds = 0
    frame_count = 0
    for n_epi in range(episodes):
        state = env.reset()
        ep_reward = 0
        state_reward = 0
        epsilon = max(0.02, 0.99 - 0.01 * (n_epi / 200))
        while True:
            frame_count += 1
            action, prob = dqn.get_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                next_state = np.empty(NUM_STATES)
                next_state.fill(np.nan)
            else:
                next_state = next_state.reshape(NUM_STATES)
            dqn.buffer.store_transition(state.reshape(NUM_STATES), action, reward, next_state)
            ep_reward += reward
            state_reward += reward * prob
            state = next_state
            if (len(dqn.buffer) >= MEMORY_START_SIZE) and (frame_count % 20 == 0):
                dqn.update(1)
            if frame_count % Q_NETWORK_ITERATION == 0:
                dqn.target_net.load_state_dict(dqn.eval_net.state_dict())
            if done:
                break

        if n_epi % print_every == 0:
            print(
                f"episode: {n_epi}, ",
                f"reward: {ep_reward}, ",
                f"state reward: {state_reward}, ",
                f"buffer size : {len(dqn.buffer)}, ",
                f"epsilon : {epsilon * 100:.1f}%",
            )
        all_rewards.append(ep_reward)
        state_rewards.append(state_reward)

        if n_epi % save_every == 0:
            torch.save(dqn.target_net.state_dict(), f'target_net_{n_epi}.pt')
            with open('all_rewards.json', 'w') as f:
                json.dump(all_rewards, f, indent=2)
            with open('state_rewards.json', 'w') as f:
                json.dump(state_rewards, f, indent=2)

        if ep_reward > 500:
            break
    return all_rewards, state_rewards


if __name__ == '__main__':
    torch.cuda.empty_cache()
    start = time.time()
    all_scores, state_rewards = train()
    end = time.time()
    print(f"Total time: {end - start} sec")
