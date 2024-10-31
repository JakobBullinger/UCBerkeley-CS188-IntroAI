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
        newFood = currentGameState.getFood()  # Note: Edited from successor state to current state
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
    
        # Get Pacman position
        xPacman, yPacman = newPos

        # Get position of nearest food
        listDistanceToFood = []
        for foodPosition in newFood.asList():
            xFood, yFood = foodPosition
            distanceToFood = abs(xPacman - xFood) + abs(yPacman - yFood)
            listDistanceToFood.append(distanceToFood)
        distanceToNearestFood = min(listDistanceToFood) if len(listDistanceToFood) > 0 else 0

        # Get position of nearest ghost
        listDistanceToGhost = []
        for ghostPosition in successorGameState.getGhostPositions():
            xGhost, yGhost = ghostPosition
            distanceToGhost = abs(xPacman - xGhost) + abs(yPacman - yGhost)
            listDistanceToGhost.append(distanceToGhost)
        distanceToNearestGhost = min(listDistanceToGhost) if len(listDistanceToGhost) > 0 else 0

        # Calculate score of successor position
        utilityDistianceToGhost = -100 if distanceToNearestGhost <= 1 else 0
        successorStateScore = 1 / (1 + distanceToNearestFood) + utilityDistianceToGhost

        return successorStateScore


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

        _, action = self.getMaxValue(gameState=gameState, agentIndex=0, depth=1)
        return action

    def getValue(self, gameState: object, agentIndex: int, depth: int):
        agentIndex += 1
        nextAgentIndex = agentIndex % gameState.getNumAgents()
        if nextAgentIndex == 0:
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState), None
        if nextAgentIndex == 0:
            return self.getMaxValue(gameState=gameState, agentIndex=nextAgentIndex, depth=depth)
        if nextAgentIndex > 0:
            return self.getMinValue(gameState=gameState, agentIndex=nextAgentIndex, depth=depth)

    def getMaxValue(self, gameState: object, agentIndex: int, depth: int):
        value = float('-inf')
        action = None
        legalActionsList = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActionsList:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            valueSuccessor, _ = self.getValue(gameState=successorGameState, agentIndex=agentIndex, depth=depth)
            if valueSuccessor > value:
                value = valueSuccessor
                bestAction = action
        return value, bestAction
    
    def getMinValue(self, gameState: object, agentIndex: int, depth: int):
        value = float('inf')
        action = None
        legalActionsList = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActionsList:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            valueSuccessor, _ = self.getValue(gameState=successorGameState, agentIndex=agentIndex, depth=depth)
            if valueSuccessor < value:
                value = valueSuccessor
                bestAction = action
            value = min(value, valueSuccessor)
        return value, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"

        _, action, _, _ = self.getMaxValue(gameState=gameState, agentIndex=0, depth=1, alpha=float('-inf'), beta=float('inf'))
        return action

    def getValue(self, gameState: object, agentIndex: int, depth: int, alpha: float, beta: float):
        agentIndex += 1
        nextAgentIndex = agentIndex % gameState.getNumAgents()
        if nextAgentIndex == 0:
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState), None, alpha, beta
        if nextAgentIndex == 0:
            return self.getMaxValue(gameState=gameState, agentIndex=nextAgentIndex, depth=depth, alpha=alpha, beta=beta)
        if nextAgentIndex > 0:
            return self.getMinValue(gameState=gameState, agentIndex=nextAgentIndex, depth=depth, alpha=alpha, beta=beta)

    def getMaxValue(self, gameState: object, agentIndex: int, depth: int, alpha: float, beta: float):
        value = float('-inf')
        bestAction = None
        legalActionsList = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActionsList:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            valueSuccessor, _, alpha, _ = self.getValue(gameState=successorGameState, agentIndex=agentIndex, depth=depth, alpha=alpha, beta=beta)
            
            if valueSuccessor > value:
                value = valueSuccessor
                bestAction = action
                alpha = max(alpha, value)
            if value > beta:
                return value, bestAction, alpha, beta

        return value, bestAction, alpha, beta
    
    def getMinValue(self, gameState: object, agentIndex: int, depth: int, alpha: float, beta: float):
        value = float('inf')
        bestAction = None
        legalActionsList = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActionsList:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            valueSuccessor, _, _, beta = self.getValue(gameState=successorGameState, agentIndex=agentIndex, depth=depth, alpha=alpha, beta=beta)

            if valueSuccessor < value:
                value = valueSuccessor
                bestAction = action
                beta = min(beta, value)
            if value < alpha:
                return value, bestAction, alpha, beta
              
        return value, bestAction, alpha, beta


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

        _, action = self.getMaxValue(gameState=gameState, agentIndex=0, depth=1)
        return action

    def getValue(self, gameState: object, agentIndex: int, depth: int):
        agentIndex += 1
        nextAgentIndex = agentIndex % gameState.getNumAgents()
        if nextAgentIndex == 0:
            depth += 1
        if gameState.isWin() or gameState.isLose() or depth > self.depth:
            return self.evaluationFunction(gameState), None
        if nextAgentIndex == 0:
            return self.getMaxValue(gameState=gameState, agentIndex=nextAgentIndex, depth=depth)
        if nextAgentIndex > 0:
            return self.getExpValue(gameState=gameState, agentIndex=nextAgentIndex, depth=depth)

    def getMaxValue(self, gameState: object, agentIndex: int, depth: int):
        value = float('-inf')
        bestAction = None
        legalActionsList = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActionsList:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            valueSuccessor, _ = self.getValue(gameState=successorGameState, agentIndex=agentIndex, depth=depth)
            
            if valueSuccessor > value:
                value = valueSuccessor
                bestAction = action

        return value, bestAction
    
    def getExpValue(self, gameState: object, agentIndex: int, depth: int):
        totalValue = 0
        legalActionsList = gameState.getLegalActions(agentIndex=agentIndex)
        legalActionCount = len(legalActionsList)
        for action in legalActionsList:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            valueSuccessor, _ = self.getValue(gameState=successorGameState, agentIndex=agentIndex, depth=depth)
            totalValue += valueSuccessor

        averageValue = totalValue / legalActionCount
        return averageValue, _


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: To calculate the score of a given state, we consider two elements besides
    from the score of the current state. The evaluation score is added to the score of the 
    current state.

    (1) Total manhatten distance to all food. This must be an inverse score as the lower the 
    total distance is, the better the score should be (since pacman is closer to completing)

    (2) The total manhatten distiance to all ghosts. We furthermore adjust this, by looking
    at wheter the ghosts are scared or not. If the ghosts are scared, then we want the total
    distiance to be as small as possible, and thus this add points to the score. When the 
    ghosts are not scared we don't want the ghosts to come closer and thus we lower the score.
    """

    "*** YOUR CODE HERE *"

    # Get relevant information
    pacmanPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTime = [ghostState.scaredTimer for ghostState in ghostStates]
    evaluationScore = 0
    foodPositionsList = foodPositions.asList()
    
    # Calculate score for food
    for foodItem in foodPositionsList:
        pacmanFoodDistance = manhattanDistance(pacmanPosition, foodItem)
        evaluationScore += 1.0 / (1.0 + pacmanFoodDistance)

    # Calculate score for ghost distance and scared time
    for ghostState, scaredTime in zip(ghostStates, scaredTime):
        ghostPosition = ghostState.getPosition()
        pacManGhostDistance = manhattanDistance(pacmanPosition, ghostPosition)
        if scaredTime == 0:
            evaluationScore -= 1.0 / (1.0 + pacManGhostDistance)
        else:
            evaluationScore += 1.0 / (1.0 + pacManGhostDistance)
    
    finalScore = currentGameState.getScore() + evaluationScore
    return finalScore


# Abbreviation
better = betterEvaluationFunction
