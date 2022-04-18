import numpy as np
import torch
import torch.nn.functional as F
from GROUP_067.neural_nets import Actor, Critic
from GROUP_067.memory import ReplayBuffer
import pickle

class Agent(object):
    """Agent class that handles the training of the networks and provides outputs as actions
    
        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use
    
    """

    def __init__(self, env_specs, max_action=1, pretrained=False):
        self.env_specs = env_specs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dim = self.env_specs['observation_space'].shape[0]
        action_dim = self.env_specs['action_space'].shape[0]

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        self.buffer_start = 1000
        self.it = 0
        self.pretrained = pretrained

    def load_weights(self, root_path):
        directory = root_path+'weights'
        filename = 'TD3'
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

    def save(self, filename='', directory=''):
            torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, 'TD3_'+filename))
            torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, 'TD3_'+filename))

    def act(self, curr_obs, mode='eval', noise=0.1):
        """Select an appropriate action from the agent policy
        
            Args:
                curr_obs (array): current state of environment
                noise (float): how much noise to add to acitons
                
            Returns:
                action (float): action clipped within action range
        
        """
        
        state = torch.FloatTensor(curr_obs.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        #add noise
        if noise != 0 and mode == 'train': 
            action = (action + np.random.normal(0, noise, size=self.env_specs['action_space'].shape[0]))

        #exploratory start
        if mode == 'train' and not self.pretrained and len(self.replay_buffer.storage) < self.buffer_start:
            action = self.env_specs['action_space'].sample()
            
        return action.clip(-1, 1)

    def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        #iteration
        self.it += 1
        
        self.replay_buffer.add((curr_obs, next_obs, action, reward, done))

        if len(self.replay_buffer.storage) > self.buffer_start:
          # Sample replay buffer storage
          x, y, u, r, d = self.replay_buffer.sample(batch_size)
          state = torch.FloatTensor(x).to(self.device)
          action = torch.FloatTensor(u).to(self.device)
          next_state = torch.FloatTensor(y).to(self.device)
          done = torch.FloatTensor(1 - d).to(self.device)
          reward = torch.FloatTensor(r).to(self.device)

          # Select action according to policy and add clipped noise 
          noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
          noise = noise.clamp(-noise_clip, noise_clip)
          next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

          # Compute the target Q value
          target_Q1, target_Q2 = self.critic_target(next_state, next_action)
          target_Q = torch.min(target_Q1, target_Q2)
          target_Q = reward + (done * discount * target_Q).detach()

          # Get current Q estimates
          current_Q1, current_Q2 = self.critic(state, action)

          # Compute critic loss
          critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

          # Optimize the critic
          self.critic_optimizer.zero_grad()
          critic_loss.backward()
          self.critic_optimizer.step()

          # Delayed policy updates
          if self.it % policy_freq == 0:

              # Compute actor loss
              actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

              # Optimize the actor 
              self.actor_optimizer.zero_grad()
              actor_loss.backward()
              self.actor_optimizer.step()

              # Update the frozen target models
              for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

              for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                  target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)