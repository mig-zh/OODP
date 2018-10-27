__author__ = 'Batchu Vishal'
import pygame
import sys
from pygame.constants import K_a, K_d, K_SPACE, K_w, K_s, QUIT, KEYDOWN
from .board import Board
#from ..base import base
#from ple.games import base
from ple.games.base.pygamewrapper import PyGameWrapper
import numpy as np
import os


class MonsterKong(PyGameWrapper):

    def __init__(self, map_config=None):
        """
        Parameters
        ----------
        None

        ####
        map_config: dict
            Customization of the map
            read only
        """

        self.map_config = map_config
        self.map_array = None
        if not map_config or not map_config.has_key('map_array'):
            self.height = 315
            self.width = 300
        else:
            self.map_array = map_config['map_array']
            self.height = 15 * len(self.map_array)
            self.width = 15 * len(self.map_array[0])

        actions = {
            "left": K_a,
            "right": K_d,
            "jump": K_SPACE,
            "up": K_w,
            "down": K_s
        }

        PyGameWrapper.__init__(
            self, self.width, self.height, actions=actions)
        # value of reward
        if not map_config.has_key('rewards'):
            self.rewards = {
                "positive": 5,
                "win": 50,
                "negative": -25,
                "tick": 0
            }
        else:
            self.rewards = map_config['rewards']

        self.allowed_fps = None

        self._dir = os.path.dirname(os.path.abspath(__file__))

        self.IMAGES = {
            "right": pygame.image.load(os.path.join(self._dir, 'assets/right.png')),
            "right2": pygame.image.load(os.path.join(self._dir, 'assets/right2.png')),
            "left": pygame.image.load(os.path.join(self._dir, 'assets/left.png')),
            "left2": pygame.image.load(os.path.join(self._dir, 'assets/left2.png')),
            "still": pygame.image.load(os.path.join(self._dir, 'assets/still.png'))
        }

    def init(self):
        # Create a new instance of the Board class
        self.newGame = Board(
            self.width,
            self.height,
            self.rewards,
            self.rng,
            self._dir,
            self.map_config)

        # Initialize the fireball timer
        # self.fireballTimer = 0

        # Assign groups from the Board instance that was created
        self.playerGroup = self.newGame.playerGroup
        self.wallGroup = self.newGame.wallGroup
        self.ladderGroup = self.newGame.ladderGroup

    def getScore(self):
        return self.newGame.score

    def game_over(self):
        return self.newGame.lives <= 0

    def step(self, dt):
        self.newGame.score += self.rewards["tick"]
        # This is where the actual game is run
        # Get the appropriate groups
        self.fireballGroup = self.newGame.fireballGroup
        self.coinGroup = self.newGame.coinGroup

        # Create fireballs as required, depending on the number of monsters in
        # our game at the moment
        # if self.fireballTimer == 0:
        #     self.newGame.CreateFireball(
        #         self.newGame.Enemies[0].getPosition(), 0)
        # elif len(self.newGame.Enemies) >= 2 and self.fireballTimer == 23:
        #     self.newGame.CreateFireball(
        #         self.newGame.Enemies[1].getPosition(), 1)
        # elif len(self.newGame.Enemies) >= 3 and self.fireballTimer == 46:
        #     self.newGame.CreateFireball(
        #         self.newGame.Enemies[2].getPosition(), 2)
        # self.fireballTimer = (self.fireballTimer + 1) % 70

        # Animate the coin
        # for coin in self.coinGroup:
        #     coin.animateCoin()

        # To check collisions below, we move the player downwards then check
        # and move him back to his original location
        self.newGame.Players[0].updateY(2)
        self.laddersCollidedBelow = self.newGame.Players[
            0].checkCollision(self.ladderGroup)
        self.wallsCollidedBelow = self.newGame.Players[
            0].checkCollision(self.wallGroup)
        self.newGame.Players[0].updateY(-2)

        # To check for collisions above, we move the player up then check and
        # then move him back down
        self.newGame.Players[0].updateY(-2)
        self.laddersCollidedAbove = self.newGame.Players[
            0].checkCollision(self.ladderGroup)
        self.wallsCollidedAbove = self.newGame.Players[
            0].checkCollision(self.wallGroup)
        self.newGame.Players[0].updateY(2)

        # Sets the onLadder state of the player
        self.newGame.ladderCheck(
            self.laddersCollidedBelow,
            self.wallsCollidedBelow,
            self.newGame.Players[0].checkCollision(self.wallGroup))

        for event in pygame.event.get():
            # Exit to desktop
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == KEYDOWN:
                # Get the ladders collided with the player
                self.laddersCollidedExact = self.newGame.Players[
                    0].checkCollision(self.ladderGroup)
                if (event.key == self.actions["jump"] and
                        (self.newGame.Players[0].onLadder == 0 or len(self.laddersCollidedAbove) == 0)) or (
                                event.key == self.actions["up"] and self.laddersCollidedExact and False):
                    # Set the player to move up
                    self.direction = 2
                    if self.newGame.Players[
                        0].isJumping == 0 and (self.wallsCollidedBelow or
                                                   (self.newGame.Players[0].onLadder == 0 or len(
                                                       self.laddersCollidedAbove) == 0)):
                        # We can make the player jump and set his
                        # currentJumpSpeed
                        self.newGame.Players[0].isJumping = 1
                        self.newGame.Players[0].currentJumpSpeed = 7

                if event.key == self.actions["right"]:
                    # self.newGame.Players[0].isJumping = 0
                    if self.newGame.direction != 4:
                        self.newGame.direction = 4
                        self.newGame.cycles = -1  # Reset cycles
                    self.newGame.cycles = (self.newGame.cycles + 1) % 4
                    if self.newGame.cycles < 2:
                        # Display the first image for half the cycles
                        self.newGame.Players[0].updateWH(self.IMAGES["still"], "H",
                                                         self.newGame.Players[0].getSpeed(), 15, 15)
                    else:
                        # Display the second image for half the cycles
                        self.newGame.Players[0].updateWH(self.IMAGES["still"], "H",
                                                         self.newGame.Players[0].getSpeed(), 15, 15)
                    wallsCollidedExact = self.newGame.Players[
                        0].checkCollision(self.wallGroup)
                    if wallsCollidedExact:
                        # If we have collided a wall, move the player back to
                        # where he was in the last state
                        self.newGame.Players[0].updateWH(self.IMAGES["still"], "H",
                                                         -self.newGame.Players[0].getSpeed(), 15, 15)
                    # self.newGame.Players[0].isJumping = 0

                if event.key == self.actions["left"]:
                    # self.newGame.Players[0].isJumping = 0
                    if self.newGame.direction != 3:
                        self.newGame.direction = 3
                        self.newGame.cycles = -1  # Reset cycles
                    self.newGame.cycles = (self.newGame.cycles + 1) % 4
                    if self.newGame.cycles < 2:
                        # Display the first image for half the cycles
                        self.newGame.Players[0].updateWH(self.IMAGES["still"], "H",
                                                         -self.newGame.Players[0].getSpeed(), 15, 15)
                    else:
                        # Display the second image for half the cycles
                        self.newGame.Players[0].updateWH(self.IMAGES["still"], "H",
                                                         -self.newGame.Players[0].getSpeed(), 15, 15)
                    wallsCollidedExact = self.newGame.Players[
                        0].checkCollision(self.wallGroup)
                    if wallsCollidedExact:
                        # If we have collided a wall, move the player back to
                        # where he was in the last state
                        self.newGame.Players[0].updateWH(self.IMAGES["still"], "H",
                                                         self.newGame.Players[0].getSpeed(), 15, 15)
                    # self.newGame.Players[0].isJumping = 0

                # If we are on a ladder, then we can move up
                if event.key == self.actions[
                        "up"] and (self.newGame.Players[0].onLadder or
                                   len(self.laddersCollidedAbove) > 0):
                    # self.newGame.Players[0].isJumping = 0
                    self.newGame.Players[0].updateWH(self.IMAGES["still"], "V",
                                                     -self.newGame.Players[0].getSpeed() , 15, 15)
                    self.laddersCollidedE = self.newGame.Players[0].checkCollision(self.ladderGroup)
                    self.wallsCollidedE = self.newGame.Players[0].checkCollision(self.wallGroup)
                    self.newGame.Players[0].setCenter(self.newGame.Players[0].getPosition())

                    
                    if len(self.wallsCollidedE) != 0 and \
                            self.wallsCollidedE[0].getPosition()[1] < self.newGame.Players[0].getPosition()[1]:
                        self.currentJumpSpeed = 0
                        self.newGame.Players[0].setPosition((self.newGame.Players[0].getPosition()[0], self.wallsCollidedE[
                            0].getPosition()[1] + (self.newGame.Players[0].height + 0)))
                        self.newGame.Players[0].setCenter(self.newGame.Players[0].getPosition())
                        self.laddersCollidedE = self.newGame.Players[0].checkCollision(self.ladderGroup)
                        self.wallsCollidedE = self.newGame.Players[0].checkCollision(self.wallGroup)


                    if len(self.laddersCollidedE) == 0 or len(self.wallsCollidedE) != 0:
                        self.newGame.Players[0].updateY(2)
                        self.laddersCollidedB = self.newGame.Players[
                            0].checkCollision(self.ladderGroup)
                        self.wallsCollidedB = self.newGame.Players[
                            0].checkCollision(self.wallGroup)
                        self.newGame.Players[0].updateY(-2)
                        if not (len(self.laddersCollidedB) != 0 and len(self.wallsCollidedB) == 0 and len(self.wallsCollidedE) == 0):
                            self.newGame.Players[0].updateWH(self.IMAGES["still"], "V",self.newGame.Players[0].getSpeed(), 15, 15)

                # If we are on a ladder, then we can move down
                if event.key == self.actions[
                        "down"] and self.newGame.Players[0].onLadder:
                    # self.newGame.Players[0].isJumping = 0
                    self.newGame.Players[0].updateWH(self.IMAGES["still"], "V",
                                                     self.newGame.Players[0].getSpeed() , 15, 15)


                    self.laddersCollidedE = self.newGame.Players[0].checkCollision(self.ladderGroup)
                    self.wallsCollidedE = self.newGame.Players[0].checkCollision(self.wallGroup)
                    self.newGame.Players[0].setCenter(self.newGame.Players[0].getPosition())
                    if len(self.wallsCollidedE) != 0 and \
                            self.wallsCollidedE[0].getPosition()[1] > self.newGame.Players[0].getPosition()[1]:
                        self.currentJumpSpeed = 0
                        self.newGame.Players[0].setPosition((self.newGame.Players[0].getPosition()[0], self.wallsCollidedE[
                            0].getPosition()[1] - (self.newGame.Players[0].height + 0)))
                        self.newGame.Players[0].setCenter(self.newGame.Players[0].getPosition())


        # Update the player's position and process his jump if he is jumping
        self.newGame.Players[0].continuousUpdate(
            self.wallGroup, self.ladderGroup, self.laddersCollidedAbove)

        '''
        We use cycles to animate the character, when we change direction we also reset the cycles
        We also change the direction according to the key pressed
        '''

        # Redraws all our instances onto the screen
        self.newGame.redrawScreen(self.screen, self.width, self.height)

        # Update the fireball and check for collisions with player (ie Kill the
        # player)
        self.newGame.fireballCheck()

        # Check for collisions between player and spikes (ie kill the player)
        self.newGame.spikeCheck()

        # Update all the monsters
        for enemy in self.newGame.Enemies:
            enemy.continuousUpdate(self.wallGroup, self.ladderGroup)
        #self.newGame.enemyCheck()

        # Collect a coin
        coinsCollected = pygame.sprite.spritecollide(
            self.newGame.Players[0], self.coinGroup, True)
        self.newGame.coinCheck(coinsCollected)

        # Check if you have reached the princess
        self.newGame.checkVictory()
        #print self.newGame.Players[0].getPosition()

