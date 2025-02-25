# Wordle-Puzzle 

This project implements a reinforcement learning agent that learns to play the Wordle game. The test code was developed with the assistance of Google Gemini AI. I leveraged Gemini's capabilities to analyze the test requirements, generate initial code, and iteratively enhance the solution using machine learning to improve the Wordle-solving strategy. 

## Workflow

The following diagram outlines the workflow of the project:
```markdown
+---------------------+
|       Start         |
+---------------------+
|
V
+---------------------+
|  Load Word List    |
+---------------------+
|
V
+---------------------+
| Load/Train Word2Vec|
|       Model        |
+---------------------+
|
V
+---------------------+
| Create Environment |
+---------------------+
|
V
+---------------------+
|     Train Agent    |
+---------------------+
|
V
+---------------------+
|      Play Game     |
+---------------------+
|
V
+---------------------+
|        End         |
+---------------------+
```

**Detailed Steps:**

1. **Start:** The process begins.

2. **Load Word List:** Load the list of possible words from the `words.txt` file.

3. **Load/Train Word2Vec Model:**
    * Check if a pre-trained Word2Vec model exists.
    * If it exists, load the model.
    * If it doesn't exist, train a new Word2Vec model on the word list and save it.

4. **Create Environment:** Initialize the `WordleEnv` with the word list and the Word2Vec model.

5. **Train Agent:**
    * Initialize the `WordleAgent` and other training components (optimizer, loss function, replay buffer).
    * Loop through episodes:
        * Reset the environment.
        * Loop through steps until the game is done:
            * Select an action (word guess) using the epsilon-greedy strategy.
            * Take a step in the environment (make the guess).
            * Observe the next state, reward, and whether the game is done.
            * Store the experience in the replay buffer.
            * Sample a batch of experiences from the replay buffer.
            * Update the agent's network using Q-learning with the sampled experiences.
        * Decay the exploration rate (`epsilon`).
        * (Periodically) Save the trained model.

6. **Play Game:**
    * Initialize the environment for playing the game.
    * Loop through steps until the game is done:
        * Get the current state from the environment.
        * Use the trained agent to select the best action (word guess).
        * Take a step in the environment (make the guess).
        * Observe the next state, reward, and whether the game is done.
        * Print the guess made by the agent.
    * If the game is done:
        * Print whether the agent won or lost.
        * Print the secret word.
        * Reset the environment and start a new game.

7. **End:** The process ends.

## Project structure
**File Descriptions:**

* **`shared/environment.py`:** Defines the `WordleEnv` class, which simulates the Wordle game environment.
* **`shared/models.py`:** Defines the `WordleAgent` class, which represents the AI agent.
* **`shared/replay_buffer.py`:** Defines the `ReplayBuffer` class, which stores the agent's experiences for experience replay.
* **`shared/services.py`:** Handles API requests and responses for interacting with the Wordle API.
  
* **`machine_learning/train_model.py`:** Contains the code for training the AI agent.
* **`train_model_main.py`:** The main script for training the model and playing a simulated game.
  
* **`play_daily_main.py`:** The main script for using the trained model to play the daily Wordle challenge.
  
* **`words.txt`:** List of words (5 letters) for training and playing game.
  
* **`wordle_agent_v3.pth`:** using the trained model without having to retrain it every time. It represents the culmination of the agent's learning process and allows you to leverage its acquired knowledge to solve Wordle puzzles.
  
* **`word2vec.model`:** stores the trained Word2Vec model you're using to generate word embeddings for your Wordle-playing agent.


## Usage

**Please follow instructions of "setup.readme.md" to install the necessary dependencies and get to know how to run the project**

