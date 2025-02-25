import random
from collections import Counter
import torch
from .services import WordleService

class WordleEnv:
    def __init__(self, word_list, word2vec_model, service):
        self.word_list = word_list
        self.word2vec_model = word2vec_model
        self.service = service
        self.secret_word = None
        self.known_correct = None
        self.known_present = None
        self.known_absent = None
        self.present_positions = None
        self.tried_words = None
        self.guess_count = 0
        self.max_guesses = 6

    def reset(self):
        self.secret_word = random.choice(self.word_list)
        self.known_correct = [""] * 5
        self.known_present = set()
        self.known_absent = set()
        self.present_positions = {}
        self.tried_words = set()
        self.guess_count = 0
        return self._get_state()

    def _encode_letter(self, letter):
        encoding = [0] * 26
        encoding[ord(letter) - ord('a')] = 1
        return encoding

    def _get_state(self):
        print(f"_get_state known_correct: {self.known_correct}")
        return self._encode_state()

    def _encode_state(self):
        state = []

        # 1. Encode known correct letters (one-hot)
        for i in range(5):
            if self.known_correct[i]:
                state.extend(self._encode_letter(self.known_correct[i]))
            else:
                state.extend([0] * 26)

        # 2. Encode known present letters (with positional information)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter in self.known_present:
                present_pos = self.present_positions.get(letter, set())
                for i in range(5):
                    state.append(1 if i in present_pos else 0)
            else:
                state.extend([0] * 5)

        # 3. Encode known absent letters (one-hot)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            state.append(1 if letter in self.known_absent else 0)

        # 4. Encode tried letters in each position (one-hot)
        for i in range(5):
            tried_letters = [word[i] for word in self.tried_words]
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                state.append(1 if letter in tried_letters else 0)

        # 5. Encode word embeddings
        state.extend(self.get_word_embedding(self.secret_word))

        return torch.tensor(state, dtype=torch.float32)

    def get_word_embedding(self, word):
        """Returns the Word2Vec embedding for a given word."""
        try:
            return self.word2vec_model.wv[word]
        except KeyError:
            return [0] * self.word2vec_model.vector_size

    def get_possible_words(self):
        """Returns the list of possible words based on the current game state."""
        return [
            word for word in self.word_list
            if all(
                (
                    self.known_correct is not None
                    and (self.known_correct[i] == "" or word[i] == self.known_correct[i])
                    and all(c not in self.known_absent for c in word)
                    and all(c in word for c in self.known_present)
                )
                for i in range(5)
            ) and word not in self.tried_words
        ]

    def step(self, action):
        guess = self.word_list[action]

        # Check if the word has already been tried
        if self.tried_words is not None and guess is not None and guess in self.tried_words:
            print(f"Word '{guess}' already tried. Choosing a different word.")
            for i, w in enumerate(self.word_list):
                if w not in self.tried_words:
                    action = i
                    guess = w
                    break

        
        result = self.service.guess_word(self.secret_word, guess, 5)
        self.guess_count += 1

        reward = 0
        done = False

        # Calculate the number of possible words before the guess (for reward shaping)
        possible_words_before = len(self.get_possible_words())

        if result is not None:
            if self.tried_words is not None:
                self.tried_words.add(guess)

            correct_count = 0
            present_count = 0

            for item in result:
                letter = item.guess
                if item.result == "correct":
                    self.known_correct[item.slot] = letter
                    if letter in self.known_present:
                        self.known_present.remove(letter)
                        self.present_positions[letter].discard(item.slot)
                        if not self.present_positions[letter]:
                            del self.present_positions[letter]
                    correct_count += 1
                elif item.result == "present":
                    self.known_present.add(letter)
                    if letter not in self.present_positions:
                        self.present_positions[letter] = {item.slot}
                    else:
                        self.present_positions[letter].add(item.slot)
                    present_count += 1
                elif item.result == "absent":
                    if letter not in self.known_correct and letter not in self.known_present:
                        self.known_absent.add(letter)

            # Calculate the number of possible words after the guess (for reward shaping)
            possible_words_after = len(self.get_possible_words())

            if correct_count == 5:
                reward = 100
                done = True
            elif self.guess_count >= self.max_guesses:
                reward = -50
                done = True
            else:
                # Reward shaping with penalty for repeated guesses and bonus for eliminating possibilities
                reward = correct_count * 10 + present_count * 5 - 1
                if guess in self.tried_words:
                    reward -= 10  # Penalty for repeating a guess
                reward += (possible_words_before - possible_words_after)  # Increased bonus

        else:
            print("Error making guess. Terminating episode...")
            reward = -100
            done = True
            return self._get_state(), reward, done  # Return immediately

        return self._get_state(), reward, done