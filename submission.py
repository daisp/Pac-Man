import random, util
from game import Agent
from pacman import GameState, PacmanRules, GhostRules
import numpy as np


#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """

    if gameState.isLose():
        return np.NINF
    if gameState.isWin():
        return np.inf

    pacmanPos = gameState.getPacmanPosition()
    ghostTimerFlag = 0
    minimalDistToGhost = np.inf
    numOfGhosts = len(gameState.getGhostStates())
    if numOfGhosts != 0:
        minimalDistToGhost = min(
            [util.manhattanDistance(pacmanPos, ghostPos) for ghostPos in gameState.getGhostPositions()])

        count = 0
        for i in range(numOfGhosts):
            if gameState.data.agentStates[i + 1].scaredTimer != 0:
                count += 1
        if count / float(numOfGhosts) > 0.5:
            ghostTimerFlag = 1
        elif count / float(numOfGhosts) < 0.5:
            ghostTimerFlag = 0
        else:
            ghostTimerFlag = random.randint(0, 1)

    minDistFromFood = np.inf
    currentFood = gameState.getFood()
    for x in range(currentFood.width):
        for y in range(currentFood.height):
            if currentFood[x][y] is True:
                minDistFromFood = min(minDistFromFood, util.manhattanDistance(pacmanPos, (x, y)))

    foodConsideration = 10 / minDistFromFood

    score = gameState.getScore()
    ghostConsideration = (
        -50 / minimalDistToGhost if 0 < minimalDistToGhost < 4 else 0) if ghostTimerFlag == 0 else 300 / minimalDistToGhost

    capsuleConsideration = 0
    if numOfGhosts != 0:
        DistToCapsuleList = [util.manhattanDistance(pacmanPos, capsulePos) for capsulePos in gameState.getCapsules()]
        minimalDistToCapsule = np.inf if len(DistToCapsuleList) == 0 else min(DistToCapsuleList)
        capsuleConsideration = (30 / minimalDistToCapsule if 0 < minimalDistToCapsule < 10 else 0)

    numOfWalls = gameState.hasWall(pacmanPos[0] + 1, pacmanPos[1]) \
                 + gameState.hasWall(pacmanPos[0] - 1, pacmanPos[1]) + gameState.hasWall(pacmanPos[0], pacmanPos[1] + 1) \
                 + gameState.hasWall(pacmanPos[0], pacmanPos[1] - 1)
    wallsConsideration = 1 / numOfWalls if numOfWalls != 0 else 1.5
    return score + ghostConsideration + capsuleConsideration + foodConsideration + wallsConsideration + random.randint(
        0, 1)


#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        # BEGIN_YOUR_CODE

        # Collect legal moves and successor states
        legal_moves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = []
        ghost_num = len(gameState.getGhostStates())
        for action in legal_moves:
            child_state = gameState.generateSuccessor(0, action)
            if ghost_num != 0:
                scores.append(self.rb_minimax(child_state, 1, 0, self.depth, 0, ghost_num))
            else:
                scores.append(self.rb_minimax(child_state, 0, 0, self.depth, 0, ghost_num))

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legal_moves[chosenIndex]


    def rb_minimax(self, cur_state, turn, agent, depth_limit, depth, ghost_num):
        if turn == agent:
            depth += 1
        if depth >= depth_limit or cur_state.isWin() or cur_state.isLose():
            return self.evaluationFunction(cur_state)
        if turn == agent:  # if Pacman's turn
            cur_max = np.NINF
            for action in cur_state.getLegalPacmanActions():  # iterating over children gameStates
                child_state = cur_state.generateSuccessor(turn, action)
                cur_max = max(cur_max, self.rb_minimax(child_state, (turn + 1) % (ghost_num + 1), agent, depth_limit,
                                                       depth, ghost_num))
            return cur_max
        else:  # if ghost turn
            assert turn > agent
            cur_min = np.Inf
            for action in cur_state.getLegalActions(turn):  # iterating over children gameStates
                child_state = cur_state.generateSuccessor(turn, action)
                cur_min = min(cur_min, self.rb_minimax(child_state, (turn + 1) % (ghost_num + 1), agent, depth_limit,
                                                       depth, ghost_num))
            return cur_min
            # END_YOUR_CODE


######################################################################################
# d: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE
        # Collect legal moves and successor states
        legal_moves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = []
        ghost_num = len(gameState.getGhostStates())
        for action in legal_moves:
            child_state = gameState.generateSuccessor(0, action)
            if ghost_num != 0:
                scores.append(self.rb_alphabeta(child_state, 1, 0, self.depth, 0, ghost_num, np.NINF, np.inf))
            else:
                scores.append(self.rb_alphabeta(child_state, 0, 0, self.depth, 0, ghost_num, np.NINF, np.inf))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legal_moves[chosenIndex]

    def rb_alphabeta(self, cur_state, turn, agent, depth_limit, depth, ghost_num, alpha, beta):
        if turn == agent:
            depth += 1
        if depth >= depth_limit or cur_state.isWin() or cur_state.isLose():
            return self.evaluationFunction(cur_state)
        if turn == agent:  # if Pacman's turn
            cur_max = np.NINF
            for action in cur_state.getLegalPacmanActions():  # iterating over children gameStates
                child_state = cur_state.generateSuccessor(turn, action)
                cur_max = max(cur_max, self.rb_alphabeta(child_state, (turn + 1) % (ghost_num + 1), agent, depth_limit,
                                                         depth, ghost_num, alpha, beta))
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return np.inf
            return cur_max

        else:  # if ghost turn
            assert turn > agent
            cur_min = np.Inf
            for action in cur_state.getLegalActions(turn):  # iterating over children gameStates
                child_state = cur_state.generateSuccessor(turn, action)
                cur_min = min(cur_min, self.rb_alphabeta(child_state, (turn + 1) % (ghost_num + 1), agent, depth_limit,
                                                         depth, ghost_num, alpha, beta))
                beta = min(cur_min, beta)
                if cur_min <= alpha:
                    return np.NINF
            return cur_min

            # END_YOUR_CODE


######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """

        # BEGIN_YOUR_CODE
        # Collect legal moves and successor states
        legal_moves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = []
        ghost_num = len(gameState.getGhostStates())
        for action in legal_moves:
            child_state = gameState.generateSuccessor(0, action)
            if ghost_num != 0:
                scores.append(self.rb_expectimax(child_state, 1, 0, self.depth, 0, ghost_num))
            else:
                scores.append(self.rb_expectimax(child_state, 0, 0, self.depth, 0, ghost_num))

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legal_moves[chosenIndex]

    def rb_expectimax(self, cur_state: GameState, turn: int, agent: int, depth_limit: int, depth: int, ghost_num: int):
        if turn == agent:
            depth += 1
        if depth >= depth_limit or cur_state.isWin() or cur_state.isLose():
            return self.evaluationFunction(cur_state)
        if turn == agent:  # if Pacman's turn
            cur_max = np.NINF
            for action in cur_state.getLegalPacmanActions():  # iterating over children gameStates
                child_state = cur_state.generateSuccessor(turn, action)
                cur_max = max(cur_max, self.rb_expectimax(child_state, (turn + 1) % (ghost_num + 1), agent, depth_limit,
                                                          depth, ghost_num))
            return cur_max
        else:  # if ghost turn
            assert turn > agent
            ghost_legal_moves = cur_state.getLegalActions(turn)
            assert len(ghost_legal_moves) is not 0
            expectancy = 0
            for action in ghost_legal_moves:
                child_state = cur_state.generateSuccessor(turn, action)
                expectancy += (1.0 / len(ghost_legal_moves)) * (
                    self.rb_expectimax(child_state, (turn + 1) % (ghost_num + 1), agent, depth_limit, depth, ghost_num))
            return expectancy

        # END_YOUR_CODE


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
        """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
      Your competition agent
    """

    def getAction(self, gameState):
        """
          Returns the action using self.depth and self.evaluationFunction

        """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
