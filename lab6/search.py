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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    analyzed = SearchElement.firstElement(problem)
    if analyzed.isGoalState():
        return []
    stack = util.Stack()
    visited = set()
    stack.push(analyzed)
    while not stack.isEmpty():
        analyzed = stack.pop()
        if analyzed.state not in visited:
            visited.add(analyzed.state)
            if analyzed.isGoalState():
                track = []
                while not analyzed.isFirst:
                    track.append(analyzed.action)
                    analyzed = analyzed.before
                track.reverse()
                return track
            else:
                successors = analyzed.getSuccessors()
                for successor in successors:
                    element = SearchElement(problem, successor[0], successor[1], analyzed, successor[2])
                    stack.push(element)

    track = []
    while not analyzed.isFirst:
        track.append(analyzed.action)
        analyzed = analyzed.before
    track.reverse()
    return track


def breadthFirstSearch(problem: SearchProblem):
    analyzed = SearchElement.firstElement(problem)
    if analyzed.isGoalState():
        return []
    visited = set()
    queue = util.Queue()
    queue.push(analyzed)
    while not queue.isEmpty():
        analyzed = queue.pop()
        if analyzed.state not in visited:
            visited.add(analyzed.state)
            if analyzed.isGoalState():
                track = []
                while not analyzed.isFirst:
                    track.append(analyzed.action)
                    analyzed = analyzed.before
                track.reverse()
                return track
            else:
                successors = analyzed.getSuccessors()
                for successor in successors:
                    element = SearchElement(problem, successor[0], successor[1], analyzed, successor[2])
                    queue.push(element)

    track = []
    while not analyzed.isFirst:
        track.append(analyzed.action)
        analyzed = analyzed.before
    track.reverse()
    return track


def uniformCostSearch(problem: SearchProblem):
    analyzed = SearchElement.firstElement(problem)
    if analyzed.isGoalState():
        return []
    visited = {}
    queue = util.PriorityQueue()
    queue.push(analyzed, 0)
    while not queue.isEmpty():
        analyzed = queue.pop()
        if (analyzed.state not in visited) or (analyzed.cost < visited[analyzed.state]):
            visited[analyzed.state] = analyzed.cost
            if analyzed.isGoalState():
                track = []
                while not analyzed.isFirst:
                    track.append(analyzed.action)
                    analyzed = analyzed.before
                track.reverse()
                return track
            else:
                successors = analyzed.getSuccessors()
                for successor in successors:
                    element = SearchElement(problem, successor[0], successor[1], analyzed, analyzed.cost + successor[2])
                    queue.update(element, element.cost)

    track = []
    while not analyzed.isFirst:
        track.append(analyzed.action)
        analyzed = analyzed.before
    track.reverse()
    return track


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    queue = util.PriorityQueue()
    visited = set()
    analyzed = SearchElement.firstElement(problem)
    queue.push(analyzed, 0)

    while not queue.isEmpty():
        analyzed = queue.pop()
        if analyzed.state in visited:
            continue
        visited.add(analyzed.state)

        if analyzed.isGoalState():
            track = []
            while not analyzed.isFirst:
                track.append(analyzed.action)
                analyzed = analyzed.before
            track.reverse()
            return track

        successors = analyzed.getSuccessors()
        for successor in successors:
            element = SearchElement(problem, successor[0], successor[1], analyzed, analyzed.cost + successor[2])
            queue.update(element, element.cost + heuristic(element.state, problem))
    return []


class SearchElement:
    def __init__(self, problem, state, action, before, cost, isFirst=False):
        self.problem = problem
        self.state = state
        self.action = action
        self.before = before
        self.isFirst = isFirst
        self.cost = cost

    @classmethod
    def firstElement(cls, problem):
        return cls(problem, problem.getStartState(), None, None, 0, True)

    def isGoalState(self):
        return self.problem.isGoalState(self.state)

    def getSuccessors(self):
        return self.problem.getSuccessors(self.state)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
