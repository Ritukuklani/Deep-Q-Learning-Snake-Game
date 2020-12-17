import numpy as np
import cv2

from PIL import Image
from operator import add


from collections import deque

NO_ACTION = 0
TURN_LEFT = 1
TURN_RIGHT = 2

class SnakeEnv:
    # Rewards
    VICTORY_REWARD = 100
    MOVE_PENALTY = -1
    HIT_WALL_PENALTY = -100
    HIT_BODY_PENALTY = -100
    EAT_FOOD_REWARD = 40

    # board-size
    WIDTH = 10
    HEIGHT = 10

    # environment
    OBSERVATION_SPACE_VALUES = (WIDTH, HEIGHT, 3)
    ACTION_SPACE_SIZE = 3

    # Dict
    SNAKE_BODY = 1
    SNAKE_HEAD = 2
    FOOD = 3

    # color dictionary
    COLORS = {SNAKE_BODY: (0, 255, 0),
              SNAKE_HEAD: (255, 192, 0),
              FOOD: (0, 0, 255)}

    def __init__(self):
        self.snake = deque()
        self.max_len = self.WIDTH * self.HEIGHT
        self.dx = 1
        self.dy = 0
        self.episode_step = 0
        self.food = None
        self.score = 0

    def get_random_position(self):
        return np.random.randint(0, self.WIDTH), np.random.randint(0, self.HEIGHT)

    def place_food(self):
        self.food = self.get_random_position()
        while self.food in self.snake:
            self.food = self.get_random_position()

    def reset(self):
        self.snake.clear()
        self.snake.append((self.WIDTH//2, self.HEIGHT//2))
        self.snake.append((self.WIDTH//2 + 1, self.HEIGHT//2))
        self.place_food()
        self.dx = 1
        self.dy = 0
        self.episode_step = 0
        self.score = 0
        return np.array(self.get_image())

    def boundary_check(self, position):
        return 0 <= position[0] < self.WIDTH and 0 <= position[1] < self.HEIGHT

    def move(self, action):
        self.episode_step += 1
        reward, done = self.MOVE_PENALTY, self.episode_step == 200
        if action == TURN_LEFT:
            self.dx, self.dy = self.dy, -self.dx
        elif action == TURN_RIGHT:
            self.dx, self.dy = -self.dy, self.dx
        snake_head = self.snake[-1]
        new_snake_head = tuple(map(add, snake_head, (self.dx, self.dy)))
        if new_snake_head in self.snake:
            reward, done = self.HIT_BODY_PENALTY, True
        if not self.boundary_check(new_snake_head):
            reward, done = self.HIT_WALL_PENALTY, True
        self.snake.append(new_snake_head)
        if new_snake_head == self.food:
            self.score += 1
            self.episode_step = 0 #LR
            reward = self.EAT_FOOD_REWARD
            self.place_food()
        else:
            self.snake.popleft()
        if len(self.snake) == self.max_len:
            reward, done = self.VICTORY_REWARD, True
        return np.array(self.get_image()), reward, done

    def get_image(self):
        image = np.zeros((self.WIDTH, self.HEIGHT, 3), dtype=np.uint8)
        for snake_body in self.snake:
            if self.boundary_check(snake_body):
                image[snake_body] = self.COLORS[self.SNAKE_BODY]
        snake_head = self.snake[-1]
        if self.boundary_check(snake_head):
            image[snake_head] = self.COLORS[self.SNAKE_HEAD]
        image[self.food] = self.COLORS[self.FOOD]
        #image[self.score]=self.COLORS[self.FOOD]
        return Image.fromarray(image, 'RGB')

    def render(self):
        image = self.get_image()
        image = image.resize((self.WIDTH * 20, self.HEIGHT * 20))
        cv2.imshow(f"image", np.array(image))
        cv2.waitKey(30)
