import gym
import matplotlib.pyplot as plt
import numpy as np

## Pos_space and vel_space are continous state spaces that show the possibilites that the agent can be or take. 
## These state spaces are pushed into discrete states by bucketing each continuous variable into multiple discrete states.
pos_space = np.linspace(-1.2, 0.6, 12)
vel_space = np.linspace(-0.07, 0.07, 20)

# Takes an observation of the environment and returns a state.
def get_state(observation):
    pos, vel =  observation
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return (pos_bin, vel_bin)

# Finding the maximum action.
def max_action(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state,a] for a in actions])            # Transitioning from a dictionary represantion to a Numpy array represantation.
    action = np.argmax(values)                                  # Then the action that the agent will take will have the highest value in the array that was just created.

    return action

# Initializing the environment.
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    n_games = 50000
    alpha = 0.1         # Learning rate
    gamma = 0.99        # Discount rate
    eps = 1             # Epsilon

    # Actions space
    action_space = [0, 1, 2]

    # States space
    states = []
    for pos in range(13):
        for vel in range(21):
            states.append((pos, vel))

    # Initializing Q
    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0
    
    score = 0                               # Initializing the score
    total_rewards = np.zeros(n_games)       # Defining the total rewards, and return a new array, filled with zeros for the number of games.
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs)
        if i % 100 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f' % eps)             # Printing the reward and epsilon for every one hundred episodes to see how the agent is learning overtime.
        score = 0
        while not done:            
            action = np.random.choice([0,1,2]) if np.random.random() < eps \
                    else max_action(Q, state)                                       # Take a random action if that random number is less than epsilon. Otherwise, take the maximum action. This is what is called Epsilon Greedy.
            obs_, reward, done, info = env.step(action)                             # Excecute that action.
            state_ = get_state(obs_)                                                # Get the new state.
            score += reward                                                         # Keep track of our score.
            action_ = max_action(Q, state_)                                         # Calculating the maximum action for the new state.
            Q[state, action] = Q[state, action] + \
                    alpha*(reward + gamma*Q[state_, action_] - Q[state, action])    # Q-Learning equation.
            state = state_                                                          # The current state will be equal to the new state
        total_rewards[i] = score
        eps = eps - 2/n_games if eps > 0.01 else 0.01                               # Epsilon decay will be linear.

    # Plotting the performance of the agent over time.
    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])                # An average of the previous fifty games.
    plt.plot(mean_rewards)
    plt.savefig('mountaincar.png')

