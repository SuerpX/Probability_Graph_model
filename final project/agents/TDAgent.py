from agents import agent
import numpy
from random import randint, uniform
from gameboard import gameboard
import math

class TDAgent(agent):
    def __init__(self, gb, sn, sl):
        super().__init__(gb)
        self.e = 0.3
        self.alpha = 0.1
        self.qtable = {}
        self.beta = 0.99
        self.sn = sn
        self.sl = sl

    def greedyPolicy(self, lactions, state):
 #       lactions = self.legalAction(state)
        maxV = float("-inf")
        maxA = -1
        ss = state.tostring()
        for a in lactions:
            v = self.qtable[ss][a]
            if v > maxV:
                maxV = v
                maxA = a
        return maxA
    
    def exploreOrExploitPolicy(self, lactions, state):
 #       lactions = self.legalAction(state)
        p = uniform(0, 1)
        if p < self.e:
            return lactions[randint(0, len(lactions) - 1)]
        else:
            return self.greedyPolicy(lactions, state)
        
    def legalAction(self, state):
        la = ['u', 'd', 'l', 'r']
        for a in self.actions:
            tempgb = gameboard(4, numpy.copy(state))
            changed = tempgb.takeAction(a)
            if not changed:
                la.remove(a)
        return la

    def action(self):
        self.qtable = {}
 #       visited = set()
        s = self.gb.board.tostring()
        self.la = self.legalAction(self.gb.board)
        self.qtable[s] = {}
        for a in self.la:
            self.qtable[s][a] = 0
        for i in range(self.sn):
            self.learning()
            
        return self.greedyPolicy(self.la, self.gb.board)
class QLAgent(TDAgent):
    def __init__(self, gb, sn, sl):
        super().__init__(gb, sn, sl)
        
    def learning(self):
        steps = 0
        tempgb = gameboard(4, numpy.copy(self.gb.board))
        while not tempgb.islost and steps < self.sl:
            la = self.legalAction(tempgb.board)
            ca = self.exploreOrExploitPolicy(la, tempgb.board)
            cs = tempgb.board.tostring()
 #           if cs not in self.qtable:
#                self.qtable[cs][ca] = 0
            lastscore = tempgb.score
            tempgb.takeAction(ca)
            if tempgb.islost:
                self.qtable[cs][ca] = self.qtable[cs][ca] + self.alpha * (-1000 - self.qtable[cs][ca])
                break
            ns = tempgb.board.tostring()
            
            if ns not in self.qtable:
                la = self.legalAction(tempgb.board)
                self.qtable[ns] = {}
                for aa in la:
                    self.qtable[ns][aa] = 0
            else:
                la = []
                for aa in self.qtable[ns]:
                    la.append(aa)
                
            r = tempgb.score - lastscore
            ga = self.greedyPolicy(la, tempgb.board)
            self.qtable[cs][ca] = self.qtable[cs][ca] + self.alpha * (r + self.beta * self.qtable[ns][ga] - self.qtable[cs][ca])
            
            steps += 1
            
class SARSAAgent(TDAgent):
    def __init__(self, gb, sn, sl):
        super().__init__(gb, sn, sl)
        
    def learning(self):
        steps = 0
        tempgb = gameboard(4, numpy.copy(self.gb.board))
        ca = self.exploreOrExploitPolicy(self.la, tempgb.board)
        while not tempgb.islost and steps < self.sl:
 #           la = self.legalAction(tempgb.board)
            cs = tempgb.board.tostring()
 #           if cs not in self.qtable:
#                self.qtable[cs][ca] = 0
            lastscore = tempgb.score
            tempgb.takeAction(ca)
            if tempgb.islost:
                self.qtable[cs][ca] = self.qtable[cs][ca] + self.alpha * (-1000 - self.qtable[cs][ca])
                break
            ns = tempgb.board.tostring()
            
            if ns not in self.qtable:
                la = self.legalAction(tempgb.board)
                self.qtable[ns] = {}
                for aa in la:
                    self.qtable[ns][aa] = 0
            else:
                la = []
                for aa in self.qtable[ns]:
                    la.append(aa)
            
            r = tempgb.score - lastscore
            na = self.exploreOrExploitPolicy(la, tempgb.board)
            self.qtable[cs][ca] = self.qtable[cs][ca] + self.alpha * (r + self.beta * self.qtable[ns][na] - self.qtable[cs][ca])
            ca = na
            steps += 1


