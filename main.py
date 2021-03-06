import random
import copy
import numpy as np
import time


class GameError(AttributeError):
    pass

class abstract:
    def ConcreteMethod(self):
        raise NotImplementedError("error message")


class Game:

    def __init__(self, n):
        self.board = []
        self.size = n
        self.half_the_size = int(n / 2)
        self.reset()

    def reset(self):
        self.board = []
        value = 'B'
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(value)
                value = self.opponent(value)
            self.board.append(row)
            if self.size % 2 == 0:
                value = self.opponent(value)

    def __str__(self):
        result = "  "
        for i in range(self.size):
            result += str(i) + " "
        result += "\n"
        for i in range(self.size):
            result += str(i) + " "
            for j in range(self.size):
                result += str(self.board[i][j]) + " "
            result += "\n"
        return result

    def valid(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def contains(self, board, row, col, symbol):
        return self.valid(row, col) and board[row][col] == symbol

    def countSymbol(self, board, symbol):
        count = 0
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == symbol:
                    count += 1
        return count

    @staticmethod
    def opponent(player):
        if player == 'B':
            return 'W'
        else:
            return 'B'

    @staticmethod
    def distance(r1, c1, r2, c2):
        return abs(r1 - r2 + c1 - c2)

    def makeMove(self, player, move):
        self.board = self.nextBoard(self.board, player, move)

    def nextBoard(self, board, player, move):
        r1 = move[0]
        c1 = move[1]
        r2 = move[2]
        c2 = move[3]
        next = copy.deepcopy(board)
        if not (self.valid(r1, c1) and self.valid(r2, c2)):
            raise GameError
        if next[r1][c1] != player:
            raise GameError
        dist = self.distance(r1, c1, r2, c2)
        if dist == 0:
            if self.openingMove(board):
                next[r1][c1] = "."
                return next
            raise GameError
        if next[r2][c2] != ".":
            raise GameError
        jumps = int(dist / 2)
        dr = int((r2 - r1) / dist)
        dc = int((c2 - c1) / dist)
        for i in range(jumps):
            if next[r1 + dr][c1 + dc] != self.opponent(player):
                raise GameError
            next[r1][c1] = "."
            next[r1 + dr][c1 + dc] = "."
            r1 += 2 * dr
            c1 += 2 * dc
            next[r1][c1] = player
        return next

    def openingMove(self, board):
        return self.countSymbol(board, ".") <= 1

    def generateFirstMoves(self, board):
        moves = []
        moves.append([0] * 4)
        moves.append([self.size - 1] * 4)
        moves.append([self.half_the_size] * 4)
        moves.append([self.half_the_size - 1] * 4)
        return moves

    def generateSecondMoves(self, board):
        moves = []
        if board[0][0] == ".":
            moves.append([0, 1] * 2)
            moves.append([1, 0] * 2)
            return moves
        elif board[self.size - 1][self.size - 1] == ".":
            moves.append([self.size - 1, self.size - 2] * 2)
            moves.append([self.size - 2, self.size - 1] * 2)
            return moves
        elif board[self.half_the_size - 1][self.half_the_size - 1] == ".":
            pos = self.half_the_size - 1
        else:
            pos = self.half_the_size
        moves.append([pos, pos - 1] * 2)
        moves.append([pos + 1, pos] * 2)
        moves.append([pos, pos + 1] * 2)
        moves.append([pos - 1, pos] * 2)
        return moves

    def check(self, board, r, c, rd, cd, factor, opponent):
        if self.contains(board, r + factor * rd, c + factor * cd, opponent) and \
                self.contains(board, r + (factor + 1) * rd, c + (factor + 1) * cd, '.'):
            return [[r, c, r + (factor + 1) * rd, c + (factor + 1) * cd]] + \
                   self.check(board, r, c, rd, cd, factor + 2, opponent)
        else:
            return []

    def generateMoves(self, board, player):
        if self.openingMove(board):
            if player == 'B':
                return self.generateFirstMoves(board)
            else:
                return self.generateSecondMoves(board)
        else:
            moves = []
            rd = [-1, 0, 1, 0]
            cd = [0, 1, 0, -1]
            for r in range(self.size):
                for c in range(self.size):
                    if board[r][c] == player:
                        for i in range(len(rd)):
                            moves += self.check(board, r, c, rd[i], cd[i], 1,
                                                self.opponent(player))
            return moves

    def playOneGame(self, p1, p2, show):
        self.reset()
        while True:
            if show:
                print(self)
                print("player B's turn")
            move = p1.getMove(self.board)
            if not move:
                # p1.losses += 1
                # p2.wins += 1
                print("Game over")
                return 'W'
            try:
                self.makeMove('B', move)
            except GameError:
                print("Game over: Invalid move by", p1.name)
                print(move)
                print(self)
                return 'W'
            if show:
                print(move)
                print(self)
                print("player W's turn")
            move = p2.getMove(self.board)
            if not move:
                # p2.losses += 1
                # p1.wins += 1
                print("Game over")
                return 'B'
            try:
                self.makeMove('W', move)
            except GameError:
                print("Game over: Invalid move by", p2.name)
                print(move)
                print(self)
                return 'B'
            if show:
                print(move)

    def playNGames(self, n, p1, p2, show):
        first = p1
        second = p2
        for i in range(n):
            print("Game", i)
            print(first.name, 'vs', second.name)
            winner = self.playOneGame(first, second, show)
            if winner == 'B':
                first.won()
                second.lost()
                print(first.name, "wins")
            else:
                first.lost()
                second.won()
                print(second.name, "wins")
            first, second = second, first
            first.side = "B"
            second.side = "W"



class Player:
    name = "Player"
    wins = 0
    losses = 0

    def results(self):
        result = self.name
        result += " Wins:" + str(self.wins)
        result += " Losses:" + str(self.losses)
        return result

    def lost(self):
        self.losses += 1

    def won(self):
        self.wins += 1

    def reset(self):
        self.wins = 0
        self.losses = 0


    def initialize(self, side):
        abstract()

    def getMove(self, board):
        abstract()


class SimplePlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Simple"

    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        n = len(moves)
        if n == 0:
            return []
        else:
            return moves[0]


class RandomPlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Random"

    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        n = len(moves)
        if n == 0:
            return []
        else:
            return moves[random.randrange(0, n)]


class HumanPlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Human"

    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        while True:
            print("Possible moves:", moves)
            n = len(moves)
            if n == 0:
                print("You must concede")
                return []
            index = input("Enter index of chosen move (0-" + str(n - 1) +
                          ") or -1 to concede: ")
            try:
                index = int(index)
                if index == -1:
                    return []
                if 0 <= index <= (n - 1):
                    print("returning", moves[index])
                    return moves[index]
                else:
                    print("Invalid choice, try again.")
            except Exception as e:
                print("Invalid choice, try again.")

class MinimaxPlayer(Game, Player):

    def __init__(self, size, depth=2, show=False):
        super(MinimaxPlayer, self).__init__(size)
        self.depth = depth - 1
        self.log_time = []
        self.side = None
        self.name = "MiniMax"
        self.show = show

    def initialize(self, side):
        self.name = "Minimax"
        self.side = side
        self.log_time = []
        self.winrate_log = []

    def getMove(self, board):
        tic = time.time()
        # board must be untouched, so deepcopy of board will be passed to decision algorithm
        action = self.MiniMaxDecision(copy.deepcopy(board))
        toc = time.time()
        self.log_time.append(toc - tic)
        if self.show:
            print(self.name, "Time:", toc - tic)
        return action


    def MiniMaxDecision(self, board):  # Analyzing available actions, returns the best action
        # return argmax MinValue(Result(board, action))
        global count
        count = 0
        actions = []
        for action in self.generateMoves(board, self.side):
            actions.append([action, self.MinValue(self.Result(board, action, self.side), 0)])

        n = len(actions)
        if n == 0:
            return []
        tmp = sorted(actions, key=self.getKey)[-1]
        if self.show:
            print("Seen States:", count)
            print("Win rate:", tmp[1])
        self.winrate_log.append(tmp[1])
        return tmp[0]

    def getKey(self, item):
        return item[1]

    def MaxValue(self, board, depth):  # returns a utility value
        side = self.side
        if self.CutoffTest(depth):
            eval = self.EvalFunc(board)
            return eval
        v = float("-inf")
        for a in self.getSuccessors(board, side):
            v = max([v, self.MinValue(self.Result(board, a, side), depth + 1)])
        return v

    def MinValue(self, board, depth):  # returns a utility value
        side = self.opponent(self.side)
        if self.CutoffTest(depth):
            eval = self.EvalFunc(board)
            return eval
        v = float("inf")
        for a in self.getSuccessors(board, side):
            v = min([v, self.MaxValue(self.Result(board, a, side), depth + 1)])
        return v

    def getSuccessors(self, board, player):  # get available actions in each situation
        return self.generateMoves(board, player)

    def Result(self, board, action, player):  # it takes an action and returns board after doing that action
        global count
        count += 1
        return self.nextBoard(board, player, action)

    def TerminalTest(self, board):  # check if game is done or not
        OwnMoves = len(self.generateMoves(board, self.side))
        OppMoves = len(self.generateMoves(board, self.opponent(self.side)))
        if OwnMoves == 0 or OppMoves == 0:
            return True
        return False

    def Utility(self, board):  # Scores terminal states
        OwnMoves = len(self.generateMoves(board, self.side))
        OppMoves = len(self.generateMoves(board, self.opponent(self.side)))
        if OwnMoves == 0 and OppMoves != 0:
            return float("-inf")
        return float("inf")

    def CutoffTest(self, depth):
        if depth >= 2:
            return True
        return False

    def EvalFunc(self, board):
        if self.TerminalTest(board):
            return self.Utility(board)
        OwnMoves = len(self.generateMoves(board, self.side))
        OppMoves = len(self.getSuccessors(board, self.opponent(self.side)))
        total_moves = OwnMoves + OppMoves
        win_rate = 100 * OwnMoves - 50 * OppMoves
        lose_rate = OppMoves / total_moves
        return win_rate * 1 + lose_rate * (-0)



class AlphaBetaPlayer(Game, Player):
    def __init__(self, size, depth=4, show=False):
        super(AlphaBetaPlayer, self).__init__(size)
        self.name = "AlphaBeta"
        self.depth = depth - 1
        self.side = None
        self.show = show
        self.EvalFunc = self.EvalFunc3
        self.log_time = []
        self.winrate_log = []


    def initialize(self, side):
        self.side = side

    def getMove(self, board):
        assert self.side is not None

        tic = time.time()
        # board must be untouched, so deepcopy of board will be passed to decision algorithm
        tmp = self.MiniMaxDecision(copy.deepcopy(board))
        toc = time.time()
        if self.show:
            print(self.name, "Time:", toc - tic)
        self.log_time.append(toc - tic)
        return tmp

    def checkEndgame(self, actions):
        n = len(actions)
        if n == 0:
            return True
        return False

    def count(self, board, color):
        counter = 0
        for row in board:
            counter += row.count(color)
        return counter

    def MiniMaxDecision(self, board):  # Analyzing available actions, returns the best action
        # return argmax MinValue(Result(board, action))
        global count
        count = 0
        ranked_actions = []

        for action in self.generateMoves(board, self.side):
            ranked_actions.append([action, self.MinValue(self.Result(board, action, self.side), float("-inf"), float("inf"), 0)])
        if self.checkEndgame(ranked_actions):
            return []

        tmp = sorted(ranked_actions, key=self.getKey)[-1]
        if self.show:
            print(self.get_round(board))
            print('Seen States:', count)
            print('Win rate:', tmp[1])
        self.winrate_log.append(tmp[1])
        return tmp[0]

    def getKey(self, item):
        return item[1]

    def MaxValue(self, board, a, b, depth):  # returns a utility value
        side = self.side
        OwnMoves = len(self.generateMoves(board, self.side))
        OppMoves = len(self.generateMoves(board, self.opponent(self.side)))
        if self.TerminalTest(OwnMoves, OppMoves):
            return self.Utility(OwnMoves, OppMoves)
        if self.CutoffTest(depth):
            return self.EvalFunc(board, OwnMoves, OppMoves)
        v = float("-inf")
        for action in self.getSuccessors(board, side):
            v = max([v, self.MinValue(self.Result(board, action, side), a, b, depth + 1)])
            if v >= b:
                return v
            a = max([a, v])
        return v

    def MinValue(self, board, a, b, depth):  # returns a utility value
        side = self.opponent(self.side)
        OwnMoves = len(self.generateMoves(board, self.side))
        OppMoves = len(self.generateMoves(board, self.opponent(self.side)))
        if self.TerminalTest(OwnMoves, OppMoves):
            return self.Utility(OwnMoves, OppMoves)
        if self.CutoffTest(depth):
            return self.EvalFunc(board, OwnMoves, OppMoves)
        v = float("inf")

        for action in self.getSuccessors(board, side):
            v = min([v, self.MaxValue(self.Result(board, action, side), a, b, depth + 1)])
            if v <= a:
                return v
            b = min([b, v])
        return v

    def getSuccessors(self, board, player):  # get available actions in each situation
        return self.generateMoves(board, player)

    def get_round(self, board):
        White = self.count(board, "W")
        Black = self.count(board, "B")
        return (self.size ** 2) - (White + Black)


    def Result(self, board, action, player):  # it takes an action and returns board after doing that action
        global count
        count += 1
        return self.nextBoard(board, player, action)

    def TerminalTest(self, OwnMoves, OppMoves):  # check if game is done or not
        if OwnMoves == 0 or OppMoves == 0:
            return True
        return False

    def Utility(self, OwnMoves, OppMoves):  # Scores terminal states
        if OwnMoves == 0 and OppMoves != 0:
            return float("-inf")
        return float("inf")

    def CutoffTest(self, depth):
        if depth >= self.depth:
            return True
        return False

    def EvalFunc1(self, board, OwnMoves, OppMoves):
        total_moves = OwnMoves + OppMoves
        # win_rate = 50 * OwnMoves - 50 * OppMoves
        # lose_rate = OppMoves / total_moves
        return 100 * (1*OwnMoves - OppMoves)/total_moves

    def EvalFunc2(self, board, OwnMoves, OppMoves):
        total_moves = OwnMoves + OppMoves
        win_rate = 100*(OwnMoves - OppMoves)/total_moves
        lose_rate = OppMoves / (OwnMoves + 1)
        if OwnMoves == OppMoves:
            win_rate = 100 * OwnMoves/total_moves
        return win_rate

    def EvalFunc3(self, board, OwnMoves, OppMoves):
        total_moves = OwnMoves + OppMoves

        if self.get_round(board) > 15:  ## It was 15
            win_rate = 100 * (1 * OwnMoves - OppMoves) / total_moves
            lose_rate = OppMoves / (OwnMoves + 1)
            if OwnMoves == OppMoves:
                win_rate = 100 * OwnMoves / total_moves
            return win_rate

        win_rate = 100 * (2 * OwnMoves - OppMoves) / total_moves
        lose_rate = OppMoves / (OwnMoves + 1)
        if OwnMoves == OppMoves:
            win_rate = 100 * OwnMoves / total_moves
        return win_rate



from matplotlib import pyplot as plt

def plot_result(p1, p2, w):
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title(p1.name)
    plt.scatter(range(len(p1.log_time)), p1.log_time)
    plt.plot(range(len(p1.log_time)), p1.log_time)

    plt.subplot(1, 2, 2)
    plt.title(p2.name)
    plt.scatter(range(len(p2.log_time)), p2.log_time)
    plt.plot(range(len(p2.log_time)), p2.log_time)

    plt.savefig('./' + "Winner is: " + w + '.png')
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.title(p1.name)
    plt.scatter(range(len(p1.winrate_log)), p1.winrate_log)
    plt.plot(range(len(p1.winrate_log)), p1.winrate_log)

    plt.subplot(1, 2, 2)
    plt.title(p2.name)
    plt.scatter(range(len(p2.winrate_log)), p2.winrate_log)
    plt.plot(range(len(p2.winrate_log)), p2.winrate_log)

    plt.savefig('./WinRate' + "Winner is: " + w + '.png')

count = 0

if __name__ == '__main__':
    n = 8
    game = Game(n)
    brain1 = AlphaBetaPlayer(n, 4, show=False)
    brain1.initialize('B')
    brain2 = RandomPlayer(n)
    brain2.initialize("W")


    game.playNGames(50, brain1, brain2, False)

    print("Alpha wins:", brain1.wins)
    print("Alpha losses:", brain1.losses)

    # winner = game.playOneGame(brain1, brain2, True)
    #
    # print("Winner is:", brain1.name if winner is "B" else brain2.name)
    # plot_result(brain1, brain2, brain1.name if winner is "B" else brain2.name)
