# from pacai.util import reflection
import pdb
from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.defense import DefensiveReflexAgent

# import random
from pacai.core import distance
from pacai.core.directions import Directions

# from pacai.student.search import uniformCostSearch
# from pacai.student.searchAgents import AnyFoodSearchProblem


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="pacai.agents.capture.dummy.DummyAgent",
    second="pacai.agents.capture.dummy.DummyAgent",
):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        OffenseAgent(firstIndex),
        DefenseAgent(secondIndex),
    ]


class minimaxCaptureAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.alpha = float("-inf")
        self.beta = float("inf")
        
        #CONSTANTS:
        self.MAX_DEPTH = 2  # Max Depth that minimax searches to
        self.A_BIG_NUMBER = 1000  # An arbitrarily really big number

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        val, action = self.alphaBeta(gameState, self.index, 0)
        return action

    def alphaBeta(self, gameState, index, depth, increaseDepth = False):
        if depth == self.MAX_DEPTH or gameState.isOver():
            return self.evaluationFunction(gameState), None
        if index in self.getTeam(gameState):
            return self.maxVal(gameState, index, depth)
        else:
            return self.minVal(gameState, index, depth, increaseDepth)

    def maxVal(self, gameState, index, depth):
        best_value = float("-inf")
        best_action = None
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            if action == Directions.STOP:
                continue
            successor = gameState.generateSuccessor(index, action)
            max_agents = gameState.getNumAgents()
            if index == max_agents - 1:
                value, _ = self.alphaBeta(successor, 0, depth, True)
            else:
                value, _ = self.alphaBeta(successor, index + 1, depth)
            if value > best_value:
                best_value = value
                best_action = action
                self.alpha = max(self.alpha, best_value)
                if value <= self.beta:
                    break
        return (best_value, best_action)

    def minVal(self, gameState, index, depth, increaseDepth):
        max_agents = gameState.getNumAgents()
        best_value = float("inf")
        best_action = None
        actions = gameState.getLegalActions(index)
        for action in actions:
            if action == Directions.STOP:
                continue
            successor = gameState.generateSuccessor(index, action)
            if (self.red and index == max_agents - 1) or increaseDepth:
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

    def evaluationFunction(self, currentGameState):
        return currentGameState.getScore()

    def getNearestFood(self, gameState, agentPos):
        """
        returns the maze distance (int) to the food closest to agentPos
        """
        closestFood = float("inf")
        for food in self.getFood(gameState).asList():
            distance = self.getMazeDistance(agentPos, food)
            if distance < closestFood:
                closestFood = distance
        return closestFood

    def getFarthestFood(self, gameState, agentPos):
        farthestFood = float("-inf")
        for food in self.getFood(gameState).asList():
            distance = self.getMazeDistance(agentPos, food)
            if distance > farthestFood:
                farthestFood = distance
        return farthestFood

    def getNearestCapsule(self, gameState, agentPos):
        closestCapsule = float("inf")
        for capsule in self.getCapsules(gameState):
            distance = self.getMazeDistance(agentPos, capsule)
            if distance < closestCapsule:
                closestCapsule = distance
        return closestCapsule

    def getEnemyAgentStates(self, gameState):
        """
        Returns a list of enemy agents' states
        """
        enemies = self.getOpponents(gameState)
        states = []
        for agent in enemies:
            states.append(gameState.getAgentState(agent))
        return states

class OffenseAgent(minimaxCaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at `pacai.core.baselineTeam` for more details about how to create an agent.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.weights = {
                'foodWeight' : 1.7, 
                'ghostWeight' : 1.2, 
                'capsuleWeight' : .5
                }

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
        oldNumFood = None
        oldState = self.getPreviousObservation()
        if oldState is not None:
            oldNumFood = self.getFood(oldState).asList()
        if oldNumFood is not None and len(numFood) < len(oldNumFood):
            foodScore += self.A_BIG_NUMBER  # Give extra points if the amount of food has gone down
        if len(numFood) > 0:  # if food is still on the map
            foodScore += 300 / len(numFood)  # less food = better score
            farthestFood = self.getFarthestFood(gameState, position)
            closestFood = self.getNearestFood(gameState, position)
            foodScore -= 1.7 * farthestFood + 3 * closestFood
        else:  # if no food then that means game finished so make that big value
            foodScore += self.A_BIG_NUMBER

        # closestCapsule = self.getNearestCapsule(gameState, position)

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
            ghostPos = gState.getPosition()
            gDistance = self.getMazeDistance(position, ghostPos)
            gScare = gState.getScaredTimer()  # get if/how long the ghost is scared
            if gScare < gDistance:  # The ghost is scared for less turns than it takes to get to
                ghostScore -= 30 / gDistance
            else:
                ghostScore += 10/gDistance
            if gDistance < 5 and gScare <= gDistance:
                if (gameState.isOnRedSide(ghostPos) and gameState.isOnBlueSide(position)) or (gameState.isOnBlueSide(ghostPos) and gameState.isOnRedSide(position)):
                    ghostScore -= 50  # Should hopefully fix it from staying in the same spot looking at an enemy ghost
                else:
                    ghostScore -= 30
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
        # Using self.getCapsules returns only the capsules on the enemy side of the board
        capsules = self.getCapsules(gameState)
        for capsule in capsules:  # more score when closer to capsules
            capScore += 15 / self.getMazeDistance(position, capsule)
        return capScore

    def evaluationFunction(self, currentGameState):
        position = currentGameState.getAgentPosition(self.index)
        foodScore = self.getFoodScore(currentGameState, position)
        ghostScore = self.getGhostScore(currentGameState, position)
        capScore = self.getCapsuleScore(currentGameState, position)
        #print(f"FoodScore: {foodScore}, ghostScore: {ghostScore}, capScore: {capScore}")
        currentGameState.addScore(
           self.weights['foodWeight'] * foodScore +
           self.weights['capsuleWeight'] * capScore +
           self.weights['ghostWeight'] * ghostScore
        )  # add all scores together
        return currentGameState.getScore()


class DefenseAgent(minimaxCaptureAgent):
    """Defends team side from enemy pacmans"""

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.MAX_DEPTH = 1

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

    def evaluationFunction(self, currentGameState):
        """
        Scoring is based on if it's in ghost or pacman form
        In ghost form, tries to chase invaders in base
        In pacman form, runs away from enemies
        If there are no invaders, it camps the an area closest
        to the nearest enemy agents
        """
        position = currentGameState.getAgentPosition(self.index)
        agentState = currentGameState.getAgentState(self.index)
        score = 0
        if self.isEnemyInBase(currentGameState):
            if agentState.isPacman():
                score = -self.A_BIG_NUMBER
            return score + self.getEnemyScore(position, currentGameState)
        else:
            score += 10
        if agentState.isPacman():
            score = -self.A_BIG_NUMBER
            score += self.getEnemyScore(position, currentGameState)
        if agentState.isGhost():
            score += 1
            score += self.getEnemyScore(position, currentGameState)
        if self.respawned(currentGameState):
        #    print("RESPAWNED")
            score = -self.A_BIG_NUMBER
        # print(f"is pacman: {agentState.isPacman()}")
        # print(
        #     f"current score {score}, eScore: {self.getEnemyScore(position, currentGameState)}, foodDef: {self.getFoodDefendingScore(currentGameState)}"
        # )
        return score

    def isEnemyInBase(self, gameState):
        """Finds nearest ghost and adds a score based on distance from ghost"""
        enemyAgents = self.getEnemyAgentStates(gameState)
        invaders = [
            a for a in enemyAgents if a.isPacman() and a.getPosition() is not None
        ]
        if len(invaders) > 0:
            return True
        return False

    def getEnemyScore(self, position, gameState):
        """Finds nearest ghost and adds a score based on distance from ghost"""
        agentState = gameState.getAgentState(self.index)
        enemyAgents = self.getEnemyAgentStates(gameState)
        invaders = [
            a for a in enemyAgents if a.isPacman() and a.getPosition() is not None
        ]
        defenders = [
            a for a in enemyAgents if a.isGhost() and a.getPosition() is not None
        ]
        score = 0
        if len(invaders) > 0:
            closestEnemyInBase = self.getClosestEnemy(position, invaders)
            score += 10 / (closestEnemyInBase + 1)
        else:
            if agentState.isGhost():
                closestEnemy = self.getClosestEnemy(position, defenders)
                score += 10 / (closestEnemy + 1)
            else:
                for defender in defenders:
                    score -= self.getMazeDistance(position, defender.getPosition())
        return score

    # def onOtherSide(self, gameState):
    #     """Lowers score to never choose to cross the middle"""
    #     defenseState = gameState.getAgentState(self.index)
    #     if self.red != gameState.isOnRedSide(defenseState.getPosition()):
    #         return True
    #     return False
    
    def getClosestEnemy(self, position, ls):
        '''
        ls: List of enemy Agent States
        position: current Position
        returns int: Distance to the closest enemy
        
        Finds the distance to the closest enemy agent in ls
        '''
        closest = float('inf')
        for item in ls:
            closest = min(
                    closest,
                    self.getMazeDistance(position, item.getPosition())
            )
        return closest

    def getFoodDefendingScore(self, gameState):
        return len(self.getFoodYouAreDefending(gameState).asList())

    def respawned(self, gameState):
        defenseState = gameState.getAgentState(self.index)
        return defenseState.getPosition() == defenseState._startPosition
