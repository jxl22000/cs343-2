# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()

        "*** YOUR CODE HERE ***"
        if len(newFood) == 0:
            return 0
        ghost_positions = successorGameState.getGhostPositions()
        for ghost in ghost_positions:
            if manhattanDistance(newPos, ghost) <= 1:
                return float('-inf') # run
        distance_to_food = [manhattanDistance(newPos, f) for f in newFood]
        nearest_food = min(distance_to_food)
        food_feature = 1 / (nearest_food * 10)
        remaining_food_penalty = -len(newFood)
        remaining_capsule_penalty = -len(successorGameState.getCapsules())

        return food_feature + remaining_food_penalty + remaining_capsule_penalty

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        best_v = float('-inf')
        best_move = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)

            v = self.get_value(successor, 1, self.depth)
            if v > best_v:
                best_v = v
                best_move = action

        return best_move
        # return self.get_value(gameState, 0, self.depth)

        util.raiseNotDefined()

    # def max_value(self, gameState: GameState, depth: int):
    #     v = float('-inf')
    #     actions = gameState.getLegalActions(0)
    #     for action in actions:
    #         successor = gameState.generateSuccessor(agentIndex=0, action=action)
    #         next = self.value(successor, 1, depth - 1)
    #         v = max(v, next)
    #         # if next > v:
    #         #     v = next
    #         #     bestAction = action
    #     return v

    # def min_value(self, gameState: GameState, depth: int, agentIndex: int):
    #     v = float('inf')

    #     # for agent in range(1, gameState.getNumAgents()):
            
    #     for action in gameState.getLegalActions(agentIndex):
    #         successor = gameState.generateSuccessor(agentIndex, action)
    #         # next = self.value(successor, 0, depth - 1)
    #         if agentIndex != gameState.getNumAgents:
    #             next = self.min_value(successor, agentIndex+1, depth)
    #         else:
    #             next = self.value(successor, 0, depth-1)

    #         v = min(v, next)
    #     return v
    
    def get_value(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # if agentIndex == 0:
        #     v = self.max_value(gameState, depth)
        #     return v
        # else:
        #     v = self.min_value(gameState, depth, agentIndex)
        #     return v

        if agentIndex==0:
            v = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                next = self.get_value(successor, 1, depth)
                v = max(v, next)
            
        else:
            v = float('inf')

            # for agent in range(1, gameState.getNumAgents()):
                
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                # next = self.value(successor, 0, depth - 1)
                if agentIndex != gameState.getNumAgents() - 1:
                    next = self.get_value(successor, agentIndex+1, depth)
                else:
                    next = self.get_value(successor, 0, depth-1)

                v = min(v, next)
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        
        best_v = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)

            v = self.get_value(successor, 1, self.depth, alpha, beta)
            if v > best_v:
                best_v = v
                best_move = action
            alpha = max(alpha, v)

        return best_move
    
        util.raiseNotDefined()
    
    def get_value(self, gameState: GameState, agentIndex: int, depth: int, alpha: int, beta: int):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex==0:
            v = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                next = self.get_value(successor, 1, depth, alpha, beta)
                v = max(v, next)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            
        else:
            v = float('inf')

            # for agent in range(1, gameState.getNumAgents()):
                
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                # next = self.value(successor, 0, depth - 1)
                if agentIndex != gameState.getNumAgents() - 1:
                    next = self.get_value(successor, agentIndex+1, depth, alpha, beta)
                else:
                    next = self.get_value(successor, 0, depth-1, alpha, beta)

                v = min(v, next)
                if v < alpha:
                    return v
                beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        best_v = float('-inf')
        best_move = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)

            v = self.get_value(successor, 1, self.depth)
            if v > best_v:
                best_v = v
                best_move = action

        return best_move
    
    def get_value(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex==0:
            v = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                next = self.get_value(successor, 1, depth)
                v = max(v, next)
            
        else:
            v = float('inf')

            # for agent in range(1, gameState.getNumAgents()):
                
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # next = self.value(successor, 0, depth - 1)
                if agentIndex != gameState.getNumAgents() - 1:
                    next = self.get_value(successor, agentIndex+1, depth)
                else:
                    next = self.get_value(successor, 0, depth-1)

                # v = min(v, next)
                v += next / len(gameState.getLegalActions(agentIndex))
            
        return v

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
