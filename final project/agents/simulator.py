from gameboard import gameboard
from random import randint

class simulator():
    def __init__(self, board = None, stepsLimited = float('inf'), beta = 1, discounted = False):
        self.sgb = gameboard(4, board = board)
        self.actions = ['u', 'd', 'l', 'r']
        self.sl = stepsLimited
        self.beta = 1
        self.d = discounted
    def takeAction(self, action):
        self.sgb.takeAction(action)
 #       return score, changed

    def simulation(self, policy):
        if not self.d:
            count = 0
            reward = 0
            while not self.sgb.islost and count < self.sl:
                a = policy()
                self.takeAction(a)
                count += 1

            if self.sgb.islost:
                reward = self.sgb.score / (self.sl - count + 1)
            else:
                reward = self.sgb.score
            return reward
        else:
            count = 0
            reward = self.sgb.score
            while not self.sgb.islost and count < self.sl:
                a = policy()
                lastScore = self.sgb.score
                self.takeAction(a)
                r = self.sgb.score - lastScore
                reward += r * self.beta
                count += 1

            if self.sgb.islost:
                reward = reward / (self.sl - count + 1)
 #           print("111")
            return reward

        
                
    def randomPolicy(self):
        return self.actions[randint(0, 3)]
