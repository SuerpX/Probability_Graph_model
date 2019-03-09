import numpy
from random import randint, uniform
from copy import deepcopy
import threading
import time
class gameboard():
    def __init__(self, n, board = None, isPrint = False):#, mutex = None):
 #       threading.Thread.__init__(self)
        self.twoPr = 0.9
        self.size = n
        self.total = n * n
        self.currentNum = 0
        self.islost = False
        self.score = 0
        self.currentAction = None
        self.isPrint = isPrint
        self.currentReward = 0
 #       self.mutex = mutex
        self.lastState = None
        if board is None:
            self.board = numpy.zeros((n,n))
 #           self.mutexp_acquire()
#            self.mutexa_acquire()
            self.generateNewNum()
#            self.mutexp_acquire()
#            self.mutexa_acquire()
            self.generateNewNum()
        else:
            self.board = board
            self.currentNum = numpy.count_nonzero(board)

    def generateNewNum(self):
        self.lastState = deepcopy(self.board)
        x = numpy.where(self.board == 0)
        if uniform(0, 1) < self.twoPr:
            num = 2
        else:
            num = 4
        index = randint(0, len(x[1]) - 1)
        self.board[x[0][index],x[1][index]] = num
        self.currentNum += 1
        if self.currentNum == self.total:
            self.islost = self.isLost()
            
            if self.islost and self.isPrint:
                print("game over")
                print(self.score)
            
#        self.mutexp_release()
        
        if self.isPrint:
            #time.sleep(0.5)
            print("\x1b[2J")
            self.printAction()
            #print("generated one new number:")
            self.printBoard(self.board)
 #           self.mutexa_release()
        
    def printAction(self):
        if self.currentAction == 'u':
            print("up:")
        elif self.currentAction == 'd':
            print("down:")
        elif self.currentAction == 'l':
            print("left:")
        elif self.currentAction == 'r':
            print("right:")
      #  if self.currentAction is not None:
       #     self.printBoard(self.lastState)
    def printBoard(self, board):
        print(self.currentNum)
        print(board)
        print("score:" + str(self.score))
        #print("")
    def takeAction(self, action):
        changed = False
        self.currentAction = action
        lastScore = self.score
        if action == 'u':
            for i in range(self.size):
                changed = self.moveCell(self.board[:,i], False) or changed
        elif action == 'd':
            for i in range(self.size):
                changed = self.moveCell(self.board[:,i], True) or changed
        elif action == 'l':
            for i in range(self.size):
                changed = self.moveCell(self.board[i,:], False) or changed
        elif action == 'r':
            for i in range(self.size):
                changed = self.moveCell(self.board[i,:], True) or changed
        '''
        if self.currentNum < self.total:
            changed = True
        '''
        if changed:
            self.generateNewNum()
        
        '''
        else:
            self.mutexp_release()
            self.mutexa_release()
        '''
        self.currentReward = self.score - lastScore
        return changed
                
    def moveCell(self, a, r):
        changed = False
        if not r:
            for i in range(1, self.size):
                if a[i] != 0:
                    index = i
                    for j in range(i - 1, -1, -1):
                        if a[j] == 0:
                            a[j], a[index], index = a[index], 0, j
                            changed = True
                        elif a[j] == a[index]:
                            temps = a[index] * 2
                            a[j], a[index] = temps, 0
                            self.score += temps
                            self.currentNum -= 1
                            changed = True
                            break
                        else:
                            break
        else:
            for i in range(self.size - 2, -1, -1):
                if a[i] != 0:
                    index = i
                    for j in range(i + 1, self.size):
                        if a[j] == 0:
                            a[j], a[index], index = a[index], 0, j
                            changed = True
                        elif a[j] == a[index]:
                            temps = a[index] * 2
                            a[j], a[index] = temps, 0
                            self.score += temps
                            self.currentNum -= 1
                            changed = True
                            break
                        else:
                            break
        
        return changed

    def isLost(self):
        for i in range(self.size):
            for j in range(self.size):
                if i > 0 and self.board[i - 1, j] == self.board[i, j]:
#                    print(1111111)
                    return False
                if i + 1 < self.size and self.board[i + 1, j] == self.board[i, j]:
#                    print(222222)
                    return False
                if j > 0 and self.board[i, j - 1] == self.board[i, j]:
#                    print(333333)
                    return False
                if j + 1 < self.size and self.board[i, j + 1] == self.board[i, j]:
#                    print(444444)
                    return False
        return True
    '''
    def getCurrentState(self):
        return numpy.copy(self.board)

    def mutexp_acquire(self):
        if self.mutex is not None:
            self.mutex[0].acquire()
    def mutexp_release(self):
        if self.mutex is not None:
            self.mutex[0].release()
            
    def mutexa_acquire(self):
        if self.mutex is not None:
            self.mutex[1].acquire()
    def mutexa_release(self):
        if self.mutex is not None:
            self.mutex[1].release()
    '''

            
'''       
g = gameboard(4)
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.generateNewNum()
g.takeAction('u')
g.takeAction('d')
g.takeAction('l')
g.takeAction('r')
'''
