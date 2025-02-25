import unittest
from shared.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.buffer = ReplayBuffer(self.capacity)

    def test_add(self):
        self.buffer.add("experience1")
        self.assertEqual(len(self.buffer), 1)

    def test_add_full_capacity(self):
        for i in range(self.capacity):
            self.buffer.add(f"experience{i}")
        self.assertEqual(len(self.buffer), self.capacity)
        self.buffer.add("new_experience")
        self.assertEqual(len(self.buffer), self.capacity)

    def test_sample(self):
        for i in range(self.capacity):
            self.buffer.add(i)  # Add numbers instead of strings
        sample = self.buffer.sample(5)
        self.assertEqual(len(sample), 5)
        for exp in sample:
            self.assertIn(exp, self.buffer.buffer) # Check if the sampled experiences are in the buffer



