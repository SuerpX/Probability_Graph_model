from agents import agent
import numpy
from random import randint, uniform
from simulator import simulator
from gameboard import gameboard
import math
class UCTAgent(agent):
    def __init__(self, gb, stepLimited, numOfSimulation, beta = 1, discounted = False):
        super().__init__(gb)
        self.sl = stepLimited
        self.nos = numOfSimulation
        self.C = 45
        self.d = discounted
        self.beta = beta
        self.qTable = {}
  #      self.test = 0
        #self.e = 0.3
        
    def action(self):

 #       self.qTable = {}
        
        lactions = self.legalAction(self.gb.board)
 #       print(lactions)
        for i in range(self.nos):
            sim = simulator(board = numpy.copy(self.gb.board), stepsLimited = self.sl, beta = self.beta, discounted = self.d)
            sboard = sim.sgb.board.tostring()
            unactions = []
            backQ = []
            for a in lactions:
                if (sboard, a) not in self.qTable:
                    unactions.append(a)
            la = lactions
 #           print(len(unactions))
            while len(unactions) == 0 and not sim.sgb.islost:
                ucbA = self.UCBPolicy(sboard, la)
                backQ.append((sboard, ucbA))
                sim.takeAction(ucbA)
                sboard = sim.sgb.board.tostring()
                la = self.legalAction(sim.sgb.board)
                unactions = []
                for a in la:
                    if (sboard, a) not in self.qTable:
                        unactions.append(a)
            if len(unactions) != 0:
                simA = unactions[randint(0, len(unactions) - 1)]
                backQ.append((sim.sgb.board.tostring(), simA))
                sim.takeAction(simA)
            r = sim.simulation(sim.randomPolicy)
   #         print(r)
 #           print(backQ)
            for backS, backA in backQ[::-1]:
                if (backS, backA) not in self.qTable:
                    self.qTable[backS, backA] = [0, 0]
                self.qTable[backS, backA][0] += r
                self.qTable[backS, backA][1] += 1
        return self.greedyPolicy(self.gb.board.tostring(), lactions)   

            
    def UCBPolicy(self, sboard, lactions):
        maxA = -1
        maxV = float('-inf')
        N = 0
        for a in lactions:
            N += self.qTable[sboard, a][1]
        for a in lactions:
            v = self.qTable[sboard, a][0] / self.qTable[sboard, a][1] + self.C * math.sqrt(math.log(N) / self.qTable[sboard, a][1])
            if v > maxV:
 #               print(v,a)
                maxV = v
                maxA = a
 #       print(maxA)
        return maxA
    
    def legalAction(self, state):
        la = ['u', 'd', 'l', 'r']
        for a in self.actions:
            tempgb = gameboard(4, numpy.copy(state))
            changed = tempgb.takeAction(a)
            if not changed:
                la.remove(a)
        return la
    def exploreOrExploitPolicy(self, lactions):
        p = uniform(0, 1)
        if p < self.e:
            return lactions[randint(0, len(lactions) - 1)]
        else:
            return self.greedyPolicy(lactions)
        
    def greedyPolicy(self, sboard, lactions):
        maxA = -1
        maxV = float('-inf')
        for a in lactions:
            v = self.qTable[sboard, a][0] / self.qTable[sboard, a][1]
            if v > maxV:
 #               print(v,a)
                maxV = v
                maxA = a
 #       print(maxV)
        return maxA
