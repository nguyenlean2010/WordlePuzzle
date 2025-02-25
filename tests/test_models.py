import unittest
import torch
from shared.models import WordleAgent

class TestWordleAgent(unittest.TestCase):

    def setUp(self):
        self.state_size = 516  # Adjust based on your state representation
        self.action_size = 1000  # Adjust based on your word list size
        self.agent = WordleAgent(self.state_size, self.action_size)

    def test_forward(self):
        state = torch.randn(self.state_size)
        output = self.agent(state)

        self.assertEqual(output.shape, (self.action_size,))