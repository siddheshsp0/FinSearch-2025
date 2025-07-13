# Following this youtube tutorial to understand Twin Delayed Deep Deterministic Policy Gradient method 
# https://www.youtube.com/watch?v=1lZOB2S17LU&t=1462s


# Imports
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.optimizers import Adam
import os


class ReplayBuffer:
    '''Handles replay buffers'''
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size # Max size of the buffer
        self.mem_cntr = 0 # Counter to track number of elements in the buffer
        # Initialising the tuple (state, action, reward, next_state, done), but in form of different 
            # numpy arrays
        self.input_shape = input_shape
        self.state_memory = np.zeros((self.mem_size, *input_shape)) # * is the unpacking operator
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool) # stores if the experience was stored at the end of an episode

    def store_transition(self, state, action, reward, next_state, done):
        '''Store new experience'''
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones


class CriticNetwork(keras.Model):
    '''Critic network (Q network)'''
    def __init__(self, fc1_dim, fc2_dim, name, chkpt_dir='tmp/td3'):
        '''fc1 and fc2 dim is dimensions of 2 fully connected layers'''
        super().__init__()
        # Initialisation
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim 
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_td3'+'.weights.h5')

        # constructing the model, 2 fully connected layers and an output
            # We dont need to specify input dimensions, it infers automatically (nice)
        self.fc1 = Dense(self.fc1_dim, activation='relu')
        self.fc2 = Dense(self.fc2_dim, activation='relu')
        self.q = Dense(1, activation=None) # Output (Q value)

    def call(self, state, action): # keras.Model function which is called when the model is called like a function
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        q1_action_value = self.fc2(q1_action_value)

        q = self.q(q1_action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dim, fc2_dim, n_actions, name, chkpt_dir='tmp/td3'):
        '''fc1 and fc2 dim is dimensions of 2 fully connected layers'''
        super().__init__()
        # Initialisation
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_td3'+'.weights.h5')

        self.fc1 = Dense(self.fc1_dim, activation='relu')
        self.fc2 = Dense(self.fc2_dim, activation='relu')
        self.action = Dense(self.n_actions, activation='tanh') # tanh activations coz
        # tanh function limits the output between -1 and 1, so we can set limits by multiplying this value
        # if needed. For example if my action is between -2 and 2, I'll multiply my action given by the 
        # model by 2

    def call(self, state):
        vals = self.fc1(state)
        vals = self.fc2(vals)

        action = self.action(vals)
        return action



# The real stuff starts here
class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, update_actor_interval=2,
                 warmup=1000, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=300, noise=0.1):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0 # for delaying the actor training so that critic can converge
        self.time_step = 0 # for the warmup procedure 
        self.warmup = warmup # initial exploration time
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval # interval at which actor is updated

        # Initialising networks
        self.actor = ActorNetwork(layer1_size, layer2_size, self.n_actions, name='actor')
        self.critic_1 = CriticNetwork(layer1_size, layer2_size, name='critic_1')
        self.critic_2 = CriticNetwork(layer1_size, layer2_size, name='critic_2')

        self.target_actor = ActorNetwork(layer1_size, layer2_size, self.n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size, name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        self.noise = noise
        self.update_network_parameters(tau=1) # set target parameters = online network parameters

    def choose_action(self, observation):
        '''Choose an action for given observation'''
        # Check if warmup period is going on
        if self.time_step < self.warmup:
            action = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else: # else, if warmup period is over
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            action = self.actor(state)[0] # Pass state through actor network, and recieve actions
        action_prime = action + np.random.normal(scale=self.noise) # Add some exploration noise to our action
        action_prime = tf.clip_by_value(action_prime, self.min_action, self.max_action) # Clip the action to allowed value
        self.time_step+=1
        return action_prime
    
    def remember(self, state, action, reward, next_state, done):
        '''Store an experience'''
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        '''Function which makes the model learn'''
        # We don't want to update parameters if atleast batch_size number of experiences are not stored,
        # else it does not make any sense to sample
        if (self.memory.mem_cntr < self.batch_size):
            return
        
        # Now we sample from our buffer for training
        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)


        # # # For Critic Network
        with tf.GradientTape(persistent=True) as tape:
            # Passing s_t+1 in the actor and computing target actions for s_t+1
                # By adding noise and then clipping it, TD3 forces the critic to learn to be robust to
                # small perturbations in action.
            noise = tf.clip_by_value(tf.random.normal(shape=[self.batch_size, self.n_actions], stddev=0.2), -0.5, 0.5)
            target_actions = self.target_actor(next_states) + noise
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)


            # Get target q values for s_t+1 and target actions for s_t+1
                # Shape is [batch_size, 1], convert it to [batch_size]
            q1_ = tf.squeeze(self.target_critic_1(next_states, target_actions), 1)
            q2_ = tf.squeeze(self.target_critic_2(next_states, target_actions), 1)

            # Get q values for s_t
            q2 = tf.squeeze(self.critic_2(states, actions), 1)
            q1 = tf.squeeze(self.critic_1(states, actions), 1)

            # Get min target q value
            critic_value_ = tf.math.minimum(q1_, q2_)

            # Computing yt (from the flowchart)
            target = rewards + self.gamma * critic_value_ * (1-dones)

            # Computing losses for the critic
            critic_1_loss = keras.losses.mean_squared_error(target, q1)
            critic_2_loss = keras.losses.mean_squared_error(target, q2)
        
        # Gradient descent !
        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        # Learn step counter update
        self.learn_step_cntr+=1

        if (self.learn_step_cntr % self.update_actor_iter != 0):
            return
        # # # For actor network
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        # Chain rule: differentiating q value wrt actor's parameters
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        # Soft update target weights (actor, critic1, critic2)
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights=weights)
        
        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_1.set_weights(weights=weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_2.set_weights(weights=weights)

    def save_model(self):
        print("Saving model...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_model(self):
        print("Building networks with dummy inputs...")

        # Create dummy input to build models
        dummy_state = tf.random.normal((1, *self.memory.input_shape))
        dummy_action = tf.random.normal((1, self.n_actions))

        # Build actor and critic networks by calling them once
        self.actor(dummy_state)
        self.critic_1(dummy_state, dummy_action)
        self.critic_2(dummy_state, dummy_action)
        self.target_actor(dummy_state)
        self.target_critic_1(dummy_state, dummy_action)
        self.target_critic_2(dummy_state, dummy_action)

        print("Loading weights...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)

