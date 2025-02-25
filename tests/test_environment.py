import unittest
from unittest.mock import MagicMock
import torch
from shared.environment import WordleEnv
from shared.services import WordleService, GuessResult

class TestWordleEnv(unittest.TestCase):

    def setUp(self):
        word_list = ["apple", "banjo", "crane"]
        mock_word2vec_model = MagicMock()
        mock_word2vec_model.vector_size = 100
        mock_service = MagicMock()
        self.env = WordleEnv(word_list, mock_word2vec_model, mock_service)
        self.env.reset()

    def _create_guess_result(self, slot, guess, result):
        """Helper function to create GuessResult objects."""
        guess_result = MagicMock()
        guess_result.slot = slot
        guess_result.guess = guess
        guess_result.result = result
        return guess_result

    def test_reset(self):
        state = self.env.reset()
        self.assertEqual(self.env.guess_count, 0)
        self.assertEqual(len(state), 416)  # Check state size

    def test_get_state(self):
        self.env.known_correct = ["a", "", "p", "", ""]
        self.env.known_present = {"l"}
        self.env.known_absent = {"b", "n"}
        self.env.present_positions = {"l": {1}}
        self.env.tried_words = {"crane"}

        state = self.env._get_state()

        self.assertEqual(len(state), 416)  # Check state size
        # Add more assertions to check specific encoding values

    def test_step_incorrect_guess(self):
        self.env.secret_word = "apple"
        self.env.service.guess_word.return_value = [
            self._create_guess_result(i, "banjo"[i], "absent") for i in range(5)  # Add self.
        ]

        next_state, reward, done = self.env.step(1)  # Guess "banjo"

        self.assertLess(reward, 0)
        self.assertFalse(done)

    def test_step_correct_guess_network_error(self):
        self.env.secret_word = "apple"
        self.env.service.guess_word.return_value = None  # Simulate network error

        next_state, reward, done = self.env.step(0)  # Guess "apple"

        self.assertLess(reward, 0)
        self.assertTrue(done)


