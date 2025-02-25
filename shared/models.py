import torch.nn as nn

class WordleAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(WordleAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        return self.fc2(x)