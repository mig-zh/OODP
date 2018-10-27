import numpy as np
import random
class NaiveAgent():
    """
            This is our naive agent. It picks actions at random!
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def pickAction(self,ob):
        return random.choice(range(self.num_actions))