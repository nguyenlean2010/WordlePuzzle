from machine_learning.train_model import load_word_list, train_agent  # Correct import
from shared.environment import WordleEnv
from shared.services import WordleService
from gensim.models import Word2Vec
import os
import torch

# --- Main Execution ---
if __name__ == "__main__":
    word_list = load_word_list()
    if word_list:
        # Load or train Word2Vec model
        word2vec_model_path = "word2vec.model"
        if os.path.exists(word2vec_model_path):
            word2vec_model = Word2Vec.load(word2vec_model_path)
            print(f"Loaded Word2Vec model from {word2vec_model_path}")
        else:
            word2vec_model = Word2Vec(sentences=[word_list], vector_size=100, window=5, min_count=1, workers=4)
            word2vec_model.save(word2vec_model_path)
            print(f"Trained Word2Vec model saved to {word2vec_model_path}")

        # Initialize the service
        service = WordleService()

        # Create the environment with the Word2Vec model and service
        env = WordleEnv(word_list, word2vec_model, service)

        # Training
        trained_agent = train_agent(env)

        # Play the game with the trained agent
        def play_wordle_with_agent(word_list, agent, word2vec_model):
            env = WordleEnv(word_list, word2vec_model, service)  # Pass word2vec_model to WordleEnv
            state = env.reset()
            done = False
            guess_count = 0
            while not done:
                guess_count += 1
                with torch.no_grad():
                    action_probs = agent(state.unsqueeze(0))
                    action = torch.argmax(action_probs).item()
                guess = word_list[action]
                next_state, reward, done = env.step(action)
                print(f"Guess {guess_count}: {guess}")
                state = next_state
                if done:
                    if reward > 0:
                        print(f"You won in {guess_count} guesses!")
                    else:
                        print(f"You lost! The word was {env.secret_word}")
                    print("Game Over, resetting")
                    state = env.reset()
                    done = False
                    guess_count = 0

        play_wordle_with_agent(word_list, trained_agent, word2vec_model)