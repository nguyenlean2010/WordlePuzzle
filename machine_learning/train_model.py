import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from shared.models import WordleAgent
from shared.environment import WordleEnv
from shared.replay_buffer import ReplayBuffer
from shared.services import WordleService
from gensim.models import Word2Vec

# --- Word List ---
def load_word_list(filename="words.txt"):
    """Loads the word list from a file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            word_list = [word.strip() for word in f.readlines()]
            word_list = [word for word in word_list if len(word) == 5]
            return word_list
    else:
        print("Word list file not found.")
        return None

# --- Training ---
def train_agent(env, num_episodes=10000, learning_rate=0.001, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                batch_size=64, buffer_capacity=10000,
                model_path="wordle_agent_v3.pth"):
    state = env.reset()
    state_size = len(state)
    agent = WordleAgent(state_size, len(env.word_list))
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))
        print(f"Loaded existing model from {model_path}")

    epsilon = epsilon_start
    replay_buffer = ReplayBuffer(buffer_capacity)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, len(env.word_list) - 1)
            else:
                with torch.no_grad():
                    q_values = agent(state.unsqueeze(0))
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Add experience to replay buffer
            replay_buffer.add((state, action, reward, next_state, done))

            # Sample from replay buffer and update network
            if len(replay_buffer) > batch_size:
                experiences = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*experiences)

                states = torch.stack(states)
                next_states = torch.stack(next_states)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Calculate Q-values
                q_values = agent(states)
                next_q_values = agent(next_states)

                # Calculate target Q-values
                target_q_values = rewards + (gamma * next_q_values.max(1).values * (1 - dones))  # Extract values

                # Calculate loss
                loss = criterion(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))

                # Update network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        # Decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Print logs for monitoring
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

        if (episode + 1) % 500 == 0:
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    torch.save(agent.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")
    return agent