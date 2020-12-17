import os
from DeepQAgent import *
from tqdm import tqdm
from snake import SnakeEnv
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
# device_count = {'GPU': 1}
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.compat.v1.disable_eager_execution()


# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
GET_STATS = 10
MODEL_SAVE = True

# Render
ISRENDER = True

# For stats
ep_rewards = [-200]
scores_history=[-100]*20000

if not os.path.isdir('models-final-check'):
    os.makedirs('models-final-check')

env = SnakeEnv()
agent = DeepQAgent(env)
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode
    #Initialise state,reward.
    episode_reward = 0 
    step = 1
    current_state = env.reset()
    done = False

    while not done:      # Using Exploitation vs Exploration ($\epsilon$-greedy strategy) to either choose a random action or  a greedy action and pre-process it for further steps.
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state)) 
        else:    
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.move(action)
        episode_reward += reward

        if ISRENDER and episode % GET_STATS == 0: #code to render while training
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done)) # Append the state to our experience replay memory 
        agent.train(done)

        current_state = new_state
        step += 1 #increment Time-step



    ep_rewards.append(episode_reward)
    if MODEL_SAVE and episode % GET_STATS == 0:
        average_reward = sum(ep_rewards[-GET_STATS:]) / len(ep_rewards[-GET_STATS:]) #stats storage
        min_reward = min(ep_rewards[-GET_STATS:])
        max_reward = max(ep_rewards[-GET_STATS:])
        agent.model.save(f'models-final-check/{MODEL_NAME}_{max_reward:_>7.2f}'
                         f'max_{average_reward:_>7.2f}'
                         f'avg_{min_reward:_>7.2f}'
                         f'min.model')

    print(f"  Episode number - : {episode}   Total Score: {env.score}") #Print score to track
    scores_history.append(env.score) #Store score


    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
