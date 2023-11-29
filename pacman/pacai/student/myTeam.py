
# from pacai.util import reflection
import pdb
from pacai.agents.capture.capture import CaptureAgent
import random
from pacai.core import distance
from pacai.core.directions import Directions
from pacai.student.search import uniformCostSearch
from pacai.student.searchAgents import AnyFoodSearchProblem


def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        DummyAgent(firstIndex),
        DummyAgent(secondIndex),
    ]


class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """
    MAX_DEPTH = 2

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.alpha = float('-inf')
        self.beta = float('inf')
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        val,action = self.alphaBeta(gameState, self.index, 0)
        return action

    def alphaBeta(self,gameState,index,depth):
        if depth == self.MAX_DEPTH or gameState.isOver():
            return self.evaluationFunction(gameState),None
        if index in self.getTeam(gameState):
            return self.maxVal(gameState, index, depth)
        else:
            return self.minVal(gameState, index, depth)

    def maxVal(self, gameState, index, depth):
        best_value = float('-inf')
        best_action = None
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            if action == Directions.STOP:
                continue
            successor = gameState.generateSuccessor(index,action)
            value, _ = self.alphaBeta(successor, index + 1, depth)
            if value > best_value:
                best_value = value
                best_action = action
                self.alpha = max(self.alpha, best_value)
                if value <= self.beta:
                    break
        return (best_value, best_action)

    def minVal(self, gameState, index, depth):
        max_agents = gameState.getNumAgents()
        best_value = float('inf')
        best_action = None
        actions = gameState.getLegalActions(index)
        for action in actions:
            if action == Directions.STOP:
                continue
            successor = gameState.generateSuccessor(index, action)
            if index == max_agents - 1:
                value, _ = self.alphaBeta(successor, 0, depth + 1)
            else:
                value, _ = self.alphaBeta(successor, index + 1, depth)
            if value < best_value:
                best_value = value
                best_action = action
                self.beta = min(value, self.beta)
                if value <= self.alpha:
                    break
        return (best_value, best_action)

    def getFoodScore(self, gameState, position):
        """
        Calculate a heuristic score based on the amount and proximity of food.

        Parameters:
        - self: The current instance of the Pacman agent.
        - gameState: The current state of the game.
        - position: The current position of the Pacman agent.

        Returns:
        - foodScore (float): The calculated heuristic score for the current game state.
          Higher scores indicate more favorable conditions.

        The function considers the number of remaining food pellets on the map and
        the proximity of the nearest food pellet to the Pacman agent. The score is
        higher when there is less food remaining on the map and when the Pacman agent
        is closer to the nearest food pellet. If no food is present, indicating that
        the game has finished, a large positive value is returned to represent a
        favorable outcome.
        """
        foodScore = 0
        numFood = self.getFood(gameState).asList()
        oldNumFood = self.getFood(self.getPreviousObservation()).asList
        if len(numFood) < len(oldNumFood):
            foodScore += 50  # add points if we made the number of pellets go down? 
        if len(numFood) > 0:  # if food is still on the map
            foodScore += 100 / len(numFood)  # less food = better score
            foodScore += 10 / self.getNearestFood(gameState, position)  # smaller bonus score based on path to nearest food
        else:  # if no food then that means game finished so make that big value
            foodScore += 1000
        return foodScore

    def getGhostScore(self, gameState, position):
        """
        Calculate a heuristic score based on the proximity and state of enemy ghosts.

        Parameters:
        - self: The current instance of the Pacman agent.
        - gameState: The current state of the game.
        - position: The current position of the Pacman agent.

        Returns:
        - ghostScore (float): The calculated heuristic score for the current game state.
          Higher scores indicate more favorable conditions.

        The function considers the positions and states of enemy ghosts. It assigns scores
        based on the proximity of ghosts to the Pacman agent and whether the ghosts are
        currently scared. A higher negative score is assigned when the Pacman agent is close
        to a non-scared ghost, and a smaller negative bonus is given for distance to scared
        ghosts. The function aims to encourage the Pacman agent to avoid non-scared ghosts
        and possibly target scared ghosts.
        """
        ghostScore = 0
        ghostStates = self.getEnemyAgentStates(gameState)
        for gState in ghostStates:  # for all the ghosts
            if not gState.isGhost():
                continue  # Ignore all non-ghosts for right now
            # TODO: Change later?

            gDistance = self.getMazeDistance(position, gState.getPosition())  # get distance to ghost
            gScare = gState.getScaredTimer() # get if/how long the ghost is scared
            if gDistance < 2:  # if next to ghost REALLY BAD unless the scare timer is long enough
                 ghostScore -= 1000 if gScare <= gDistance else -500  # to reach ghost then go for ghost
            else:  # otherwise give a smaller bonus for distance to ghosts/scared ghosts
                 ghostScore -= 10 / gDistance if gScare <= gDistance else -30 / gDistance
        return ghostScore

    def getCapsuleScore(self, gameState, position):
        """
        Calculate a heuristic score based on the proximity to power capsules.

        Parameters:
        - self: The current instance of the Pacman agent.
        - gameState: The current state of the game.
        - position: The current position of the Pacman agent.

        Returns:
        - capScore (float): The calculated heuristic score for the current game state.
          Higher scores indicate more favorable conditions.

        The function considers the positions of power capsules. It assigns scores based
        on the proximity of the Pacman agent to the power capsules. A higher score is
        assigned when the Pacman agent is closer to a power capsule, encouraging the
        agent to prioritize reaching and consuming capsules during gameplay.
        """
        capScore = 0
        capsules = self.getCapsules(gameState)  # Using self.getCapsules returns only the capsules on the enemy side of the board
        for capsule in capsules:  # more score when closer to capsules
            capScore += 15 / self.getMazeDistance(position, capsule)
        return capScore

    def evaluationFunction(self, currentGameState):
        position = currentGameState.getAgentPosition(self.index)
        foodScore = self.getFoodScore(currentGameState, position)
        ghostScore = self.getGhostScore(currentGameState, position)
        capScore = self.getCapsuleScore(currentGameState, position)
        print(f"FoodScore: {foodScore}, ghostScore: {ghostScore}, capScore: {capScore}")
        currentGameState.addScore(foodScore + capScore + ghostScore)  # add all scores together
        return currentGameState.getScore()

    def getNearestFood(self, gameState, agentPos):
        '''
        returns the maze distance (int) to the food closest to agentPos
        '''
        closestFood = float('inf')
        for food in self.getFood(gameState).asList():
            distance = self.getMazeDistance(agentPos, food)
            if distance < closestFood:
                closestFood = distance
        return closestFood

    def getEnemyAgentStates(self, gameState):
        '''
        Returns a list of enemy agents' states
        '''
        enemies = self.getOpponents(gameState)
        states = []
        for agent in enemies:
            states.append(gameState.getAgentState(agent))
        return states


class DefenseAgent(DummyAgent):
    """Defends team side from enemy pacmans"""
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

    def evaluationFunction(self, currentGameState):

        position = currentGameState.getAgentPosition(self.index)
        numFood = self.getFood(currentGameState).asList()
        # ghostStates = currentGameState.getGhostStates()
        capsules = currentGameState.getCapsules(currentGameState)

        foodScore = 0  # first score to check is food
        if len(numFood) > 0:  # if food is still on the map
            foodScore += 100 / len(numFood)  # less food = better score
            # path = uniformCostSearch(AnyFoodSearchProblem(currentGameState))
            foodScore += 10 / self.getNearestFood(currentGameState, position)  # smaller bonus score based on path to nearest food
        else:  # if no food then that means game finished so make that big value
            foodScore += 1000

        # ghostScore = 0  # second score to check is ghosts
        # for gState in ghostStates:  # for all the ghosts
        #     gDistance = distance.manhattan(position, gState.getPosition())  # get distance to ghost
        #     gScare = gState.getScaredTimer()  # get if/how long the ghost is scared
        # if gDistance < 2:  # if next to ghost REALLY BAD unless the scare timer is long enough
        #     ghostScore -= 1000 if gScare <= gDistance else -500  # to reach ghost then go for ghost
        # else:  # otherwise give a smaller bonus for distance to ghosts/scared ghosts
        #     ghostScore -= 10 / gDistance if gScare <= gDistance else -30 / gDistance

        capScore = 0  # third score to check is capsules
        for capsule in capsules:  # more score when closer to capsules
            capScore += 15 / distance.maze(capsule, position, currentGameState)



        currentGameState.addScore(foodScore + capScore)  # add all scores together
        return currentGameState.getScore()

    def getNearestFood(self, gameState, agentPos):
        closestFood = float('inf')
        for food in self.getFood(gameState).asList():
            distance = self.getMazeDistance(agentPos, food)
            if distance < closestFood:
                closestFood = distance

        return closestFood
