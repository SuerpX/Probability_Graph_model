from agents import agent
import numpy
from random import randint, uniform
from simulator import simulator
from gameboard import gameboard
import math
class MCAgent(agent):
    def __init__(self, gb, stepLimited, numOfSimulation):
        super().__init__(gb)
        self.sl = stepLimited
        self.nos = numOfSimulation
        self.e = 0.3
    def action(self):
        lactions = self.legalAction()
        self.rewardOfActions = {}
        for _ in range(self.nos):
            sim = simulator(board = numpy.copy(self.gb.board), stepsLimited = self.sl)
            unactions = []
            if len(self.rewardOfActions) != len(lactions) * 2:
                for a in lactions:
                    if (a, 't') not in self.rewardOfActions:
                        unactions.append(a)
                a = unactions[randint(0, len(unactions) - 1)]
            else:
                 a = self.exploreOrExploitPolicy(lactions)
            sim.takeAction(a)
            r = sim.simulation(sim.randomPolicy)

            if (a, 't') not in self.rewardOfActions:
                self.rewardOfActions[a, 't'] = 0
                self.rewardOfActions[a, 'c'] = 0
            self.rewardOfActions[a, 't'] += r
            self.rewardOfActions[a, 'c'] += 1
        return self.greedyPolicy(lactions)
            
    def legalAction(self):
        la = ['u', 'd', 'l', 'r']
        for a in self.actions:
            tempgb = gameboard(4, numpy.copy(self.gb.board))
            changed = tempgb.takeAction(a)
            changed
            if not changed:
                la.remove(a)
        return la
    def exploreOrExploitPolicy(self, lactions):
        p = uniform(0, 1)
        if p < self.e:
            return lactions[randint(0, len(lactions) - 1)]
        else:
            return self.greedyPolicy(lactions)
        
    def greedyPolicy(self, lactions):
        maxA = -1
        maxV = float('-inf')
        for a in lactions:
            v = self.rewardOfActions[a, 't'] / self.rewardOfActions[a, 'c']
            if v > maxV:
 #               print(v,a)
                maxV = v
                maxA = a
        return maxA
    
