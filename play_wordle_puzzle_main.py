from machine_learning.train_model import WordleAgent, load_word_list
from shared.environment import WordleEnv
from shared.services import WordleService
from gensim.models import Word2Vec
import os
import torch

# --- Play Daily Wordle ---
def play_daily_wordle(word_list, agent, word2vec_model, service):
    env = WordleEnv(word_list, word2vec_model, service)
    state = env.reset()  # Not necessary for /daily, but included for consistency
    done = False
    guess_count = 0

    while not done:
        guess_count += 1
        with torch.no_grad():
            action_probs = agent(state.unsqueeze(0))
            action = torch.argmax(action_probs).item()

        guess = word_list[action]
        result = service.guess_word(None, guess)  # Use service to make the guess against /daily

        if result:
            print(f"Guess {guess_count}: {guess}")
            next_state, reward, done = env.step(action)
            state = next_state

            if done:
                if reward > 0:
                    print(f"You won in {guess_count} guesses!")
                else:
                    print(f"You lost! The word was {env.secret_word}")
        else:
            print("Error making guess. Exiting.")
            break

# --- Main Execution ---
if __name__ == "__main__":
    word_list = load_word_list()
    if word_list:
        # Load Word2Vec model
        word2vec_model_path = "word2vec.model"
        if os.path.exists(word2vec_model_path):
            word2vec_model = Word2Vec.load(word2vec_model_path)
            print(f"Loaded Word2Vec model from {word2vec_model_path}")
        else:
            print("Word2Vec model not found. Please train the model first.")
            exit()

        # Initialize the service
        service = WordleService()

        # Load the trained agent
        model_path = "wordle_agent_v3.pth"  # Update with your model's filename
        state_size = 416 + word2vec_model.vector_size  # Adjust based on your state representation
        agent = WordleAgent(state_size, len(word_list))

        if os.path.exists(model_path):
            agent.load_state_dict(torch.load(model_path))
            print(f"Loaded trained model from {model_path}")
        else:
            print("Trained model not found. Please train the model first.")
            exit()

        # Play the daily Wordle
        play_daily_wordle(word_list, agent, word2vec_model, service)