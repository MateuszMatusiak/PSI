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
import math

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        if successorGameState.isWin():
            return 1000000

        closestFood = 1000000
        for food in newFood.asList():
            closestFood = min(manhattanDistance(newPos, food), closestFood)

        for ghost in newGhostStates:
            if manhattanDistance(ghost.getPosition(), newPos) < 2 and ghost.scaredTimer == 0:
                return -1000000
        return successorGameState.getScore() + 1.0 / closestFood


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        return self.minimax(gameState, 0, 0)[0]

    def minimax(self, state, agentIndex, depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return None, self.evaluationFunction(state)
        if agentIndex == 0:
            return self.maxValue(state, agentIndex, depth)
        else:
            return self.minValue(state, agentIndex, depth)

    def maxValue(self, state, agentIndex, depth):
        bestScore = -math.inf
        bestAction = None
        for action in state.getLegalActions(agentIndex):
            score = self.minimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[1]

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction, bestScore

    def minValue(self, state, agentIndex, depth):
        bestScore = math.inf
        bestAction = None

        for action in state.getLegalActions(agentIndex):
            if agentIndex == state.getNumAgents() - 1:
                score = self.minimax(state.generateSuccessor(agentIndex, action), 0, depth + 1)[1]
            else:
                score = self.minimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[1]

            if score < bestScore:
                bestScore = score
                bestAction = action

        return bestAction, bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        return self.alphaBetaPruning(gameState, 0, 0, -math.inf, math.inf)[0]

    def alphaBetaPruning(self, state, agentIndex, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == self.depth:
            return None, self.evaluationFunction(state)
        if agentIndex == 0:
            return self.maxValue(state, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(state, agentIndex, depth, alpha, beta)

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        bestScore = -math.inf
        bestAction = None
        for action in state.getLegalActions(agentIndex):
            score = self.alphaBetaPruning(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)[1]

            if score > bestScore:
                bestScore = score
                bestAction = action

            if score > beta:
                return bestAction, bestScore
            alpha = max(alpha, score)

        return bestAction, bestScore

    def minValue(self, state, agentIndex, depth, alpha, beta):
        bestScore = math.inf
        bestAction = None

        for action in state.getLegalActions(agentIndex):
            if agentIndex == state.getNumAgents() - 1:
                score = self.alphaBetaPruning(state.generateSuccessor(agentIndex, action), 0, depth + 1, alpha, beta)[1]
            else:
                score = self.alphaBetaPruning(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, alpha, beta)[1]

            if score < bestScore:
                bestScore = score
                bestAction = action

            if score < alpha:
                return bestAction, bestScore
            beta = min(beta, score)

        return bestAction, bestScore


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

        return self.expectiminimax(gameState, 0, 0)[0]

    def expectiminimax(self, state, agentIndex, depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return None, self.evaluationFunction(state)
        if agentIndex == 0:
            return self.maxValue(state, agentIndex, depth)
        else:
            return self.randValue(state, agentIndex, depth)

    def maxValue(self, state, agentIndex, depth):
        bestScore = -math.inf
        bestAction = None
        for action in state.getLegalActions(agentIndex):
            score = self.expectiminimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[1]

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction, bestScore

    def randValue(self, state, agentIndex, depth):
        bestScore = 0
        actions = state.getLegalActions(agentIndex)

        for action in actions:
            if agentIndex == state.getNumAgents() - 1:
                score = self.expectiminimax(state.generateSuccessor(agentIndex, action), 0, depth + 1)[1]
            else:
                score = self.expectiminimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[1]

            bestScore += score * (1 / len(actions))

        return None, bestScore


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
