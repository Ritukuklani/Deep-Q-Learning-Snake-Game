import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from ModifiedTensorBoard import ModifiedTensorBoard

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "DQNModel"
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_EVERY = 5

class DeepQAgent:
    def __init__(self, env):
        self.model = self.create_model(env)
        # self.model.load_weights('models/model_file.model')

        self.target_model = self.create_model(env)
        self.target_model.set_weights(self.model.get_weights()) #copying q model into target model(weights)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard()

        self.update_target_count = 0

    @staticmethod
    def create_model(env):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, r_m):
        self.replay_memory.append(r_m)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train(self, end_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) #Sample a random minibatch of 64 transitions in the replay memory   
        

        current_states = np.array([r_m[0] for r_m in minibatch]) / 255 #fill out with the information and values we get from the Q-_model network
        current_qs_list = self.model.predict(current_states) 

        new_states = np.array([r_m[3] for r_m in minibatch]) / 255 #use the target\_model to get the action and the max Q
        target_qs = self.target_model.predict(new_states) 

        x = []
        y = []

        for ii, (current_state, action, reward, new_state, done) in enumerate(minibatch):
        #  Update/set the value of the action we chose above in the  random minibatch to the reward given at that state  
            if not done:
                maxnext_q = np.max(target_qs[ii]) 
                new_q = reward + DISCOUNT * maxnext_q #Qval assignment
            else:
                new_q = reward #Death reward/terminate

            current_qs = current_qs_list[ii]
            current_qs[action] = new_q 

            x.append(current_state) #added as data to x and y that goes into our model
            y.append(current_qs)

            #Train the network with the new values calculated with Q-learning and update the target model weights periodically.
        self.model.fit(np.array(x) / 255, np.array(y),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if end_state else None)

        if end_state:
            self.update_target_count += 1

        if self.update_target_count > UPDATE_EVERY:
            self.target_model.set_weights(self.model.get_weights())#update weights of target model periodically
            self.update_target_count = 0

        

