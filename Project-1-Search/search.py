# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):

    """Search the deepest nodes in the search tree first."""

    from util import Stack
    stack = Stack()
    visited = set()

    position = problem.getStartState()
    route = []
    stack.push((position, route))
    visited.add(position)

    while not stack.isEmpty():
        position, route = stack.pop()

        # Check if goal state is reached
        if problem.isGoalState(position) == True:
            return route

        # Add position to visited state
        if position not in visited:
             visited.add(position)

        # Add successor and direction to queue
        for successor, direction, successorCost in problem.getSuccessors(position):
            if successor not in visited:
                newRoute = route + [direction]
                stack.push((successor, newRoute))


def breadthFirstSearch(problem: SearchProblem):

    """Search the shallowest nodes in the search tree first."""

    from util import Queue
    queue = Queue()
    visited = set()

    startPosition = problem.getStartState()
    route = []
    queue.push((startPosition, route))
    visited.add(startPosition)

    while not queue.isEmpty():
        position, route = queue.pop()

        # Check if goal state is reached
        if problem.isGoalState(position) == True:
            return route

        # Add successor and direction to queu
        for successor, direction, successorCost in problem.getSuccessors(position):
            if successor not in visited:
                visited.add(successor)
                newRoute = route + [direction]
                queue.push((successor, newRoute))


def uniformCostSearch(problem: SearchProblem):

    """Search the node of least total cost first."""

    from util import PriorityQueue
    priorityQueue = PriorityQueue()
    visited = set()

    startPosition = problem.getStartState()
    item = (startPosition, [], 0)
    priorityQueue.push(item, 0)

    while not priorityQueue.isEmpty():
        position, route, cost = priorityQueue.pop()

        # Check if goal state is reached
        if problem.isGoalState(position) == True:
            return route

        # Add position to visited state
        if position not in visited:
            visited.add(position)

            # Add successor and direction to priority queue
            for successor, successorDirection, successorCost in problem.getSuccessors(position):
                if successor not in visited:
                    newRoute = route + [successorDirection]
                    newCost = cost + successorCost
                    priorityQueue.push((successor, newRoute, newCost), newCost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    
    """Search the node that has the lowest combined cost and heuristic first."""

    from util import PriorityQueue
    priorityQueue = PriorityQueue()
    visited = set()

    startPosition = problem.getStartState()
    route =  []
    cost = 0
    item = (startPosition, route, cost)
    priorityQueue.push(item, 0)

    while not priorityQueue.isEmpty():
        position, route, cost = priorityQueue.pop()

        # Check if goal state is reached
        if problem.isGoalState(position) == True:
            return route

        # Add position to visited state
        if position not in visited:
            visited.add(position)

            # Add successor and direction to queue
            for successor, successorDirection, successorCost in problem.getSuccessors(position):
                if successor not in visited:
                    newRoute = route + [successorDirection]
                    newCost = cost + successorCost
                    heuristicDistance = heuristic(successor, problem)
                    totalCost = newCost + heuristicDistance
                    priorityQueue.push((successor, newRoute, newCost), totalCost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
