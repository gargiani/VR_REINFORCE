from methods import GPOMDP, NumericalMethod
from policies import GaussianPolicy
import torch
import gym
import numpy as np

class Environment:

    def __init__(self, render=False, seed=None):
 
       self.render = render
       self.env_seed = seed

    def set_seed(self):

        if self.set_seed is not None:
            self.env.seed(self.seed)

    def simulate(self, N, T, policy=None):

        states_n = []
        actions_n = []
        rewards_n = []

        tot_reward = 0

        for episode in range(N):
            
            print("episode {} of {}\n".format(episode+1, N))

            done = False

            states = []
            actions = []
            rewards = []

            observation = self.env.reset()
            

            while not done:
                
                if self.render:
                    self.env.render()   

                states.append(observation.tolist()) 
                
                if policy==None:
                    action = self.env.action_space.sample()
                else:
                    policy.distribution(torch.tensor([observation], dtype=torch.float32))
                    action = policy.sample()[0].numpy()

                observation, reward, done, info = self.env.step(action)
                
                tot_reward += reward

                rewards.append(reward)
                actions.append(action.tolist())

            states_n.append(states)
            actions_n.append(actions)
            rewards_n.append(rewards)

        tot_reward = tot_reward/N    
        self.env.close()
        
        return {"states": states_n, "actions": actions_n, "rewards": rewards_n}


#CartPole
class CartPole(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('CartPole-v1')
        self.state_space = ("Continuous", 4)
        self.action_space = ("Discrete", 2, [0,1])

#Pendulum
class Pendulum(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('Pendulum-v0')
        self.state_space = ("Continuous", 3)
        self.action_space = ("Continuous", 1)


#MountainCar
class MountainCar(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('MountainCarContinuous-v0')
        self.state_space = ("Continuous", 2)
        self.action_space = ("Continuous", 1)

#BipedalWalker
class BipedalWalker(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('BipedalWalker-v3')
        self.state_space = ("Continuous", 24)
        self.action_space = ("Continuous", 4)

#LunarLanderContinuous
class LunarLanderContinuous(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('LunarLanderContinuous-v2')
        self.state_space = ("Continuous", 8)
        self.action_space = ("Continuous", 2)

#LunarLander
class LunarLander(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('LunarLander-v2')
        self.state_space = ("Continuous", 8)
        self.action_space = ("Discrete", 4, [0, 1, 2, 3])


if __name__ == "__main__":

    from policies import Neural_SoftMax, neuralnet
    from methods import REINFORCE, GPOMDP

    #CARTPOLE SCENARIO
    cartpole = CartPole(render=True)

    state_size = cartpole.state_space[1]
    action_size = cartpole.action_space[1]
    
    actions = torch.tensor(cartpole.action_space[2], dtype=torch.float32)

    net = neuralnet(state_size+action_size, action_size, [32])
    softmax_policy = Neural_SoftMax(net, actions)
    
    trajectories = cartpole.simulate(10, 100, policy=softmax_policy)

    discount_factor = 0.9
    method = GPOMDP(0.001, softmax_policy.neural_net.parameters, discount_factor)

    for ii in range(10): 

        observations = torch.tensor(trajectories["states"][ii])
        actions = torch.tensor(trajectories["actions"][ii])   
        rewards = torch.tensor(trajectories["rewards"][ii])
        
        logprob = softmax_policy.log_prob(observations, actions, 10)
        method.grad_estimator(logprob, rewards)
        
    method.step()
    method.reset_grad()

    for param in softmax_policy.neural_net.parameters():
        print(param.grad)

    #PENDULUM
    pendulum = Pendulum(render=True)

    state_size = pendulum.state_space[1]
    action_size = pendulum.action_space[1]
    
    net = neuralnet(state_size, action_size, [32])
    gaussian_policy = GaussianPolicy(net, torch.eye(action_size))
    
    trajectories = pendulum.simulate(10, 100, policy=gaussian_policy)

    discount_factor = 0.9
    method = REINFORCE(0.001, gaussian_policy.mean.parameters, discount_factor)

    for ii in range(10):

        observations = torch.tensor(trajectories["states"][ii])
        actions = torch.tensor(trajectories["actions"][ii])
        rewards = torch.tensor(trajectories["rewards"][ii])

        #date le osservazioni creo la distribuzione sulle azioni
        logprob = gaussian_policy.log_prob(observations, actions, 10)
        method.grad_estimator(logprob, rewards)

    method.step()
    method.reset_grad()

    for param in gaussian_policy.mean.parameters():
        print(param.grad)

    import pdb; pdb.set_trace()
    #MOUNTAINCAR
    mountaincar = MountainCar(render=True)

    state_size = mountaincar.state_space[1]
    action_size = mountaincar.action_space[1]
    
    net = neuralnet(state_size, action_size, [32])
    gaussian_policy = GaussianPolicy(net, torch.eye(action_size))
    
    trajectories = mountaincar.simulate(10, 100, policy=gaussian_policy)

    #BIPEDALWALKER
    bipedalwalker = BipedalWalker(render=True)

    state_size = bipedalwalker.state_space[1]
    action_size = bipedalwalker.action_space[1]
    
    net = neuralnet(state_size, action_size, [32])
    gaussian_policy = GaussianPolicy(net, torch.eye(action_size))
    
    trajectories = bipedalwalker.simulate(10, 100, policy=gaussian_policy)

    discount_factor = 0.9
    method = REINFORCE(0.001, gaussian_policy.mean.parameters, discount_factor)

    for ii in range(10):

        observations = torch.tensor(trajectories["states"][ii])
        actions = torch.tensor(trajectories["actions"][ii])
        rewards = torch.tensor(trajectories["rewards"][ii])

        #date le osservazioni creo la distribuzione sulle azioni
        logprob = gaussian_policy.log_prob(observations, actions, 10)
        method.grad_estimator(logprob, rewards)

    method.step()
    method.reset_grad()

    for param in gaussian_policy.mean.parameters():
        print(param.grad)

    #LUNARLANDERCONTINUOUS
    lunarlandercontinuous = LunarLanderContinuous(render=True)

    state_size = lunarlandercontinuous.state_space[1]
    action_size = lunarlandercontinuous.action_space[1]
    
    net = neuralnet(state_size, action_size, [32])
    gaussian_policy = GaussianPolicy(net, torch.eye(action_size))

    trajectories = lunarlandercontinuous.simulate(10, 100, policy=gaussian_policy)
    
    #LUNARLANDER
    lunarlander = LunarLander(render=True)

    state_size = lunarlander.state_space[1]
    action_size = lunarlander.action_space[1]
    
    actions = torch.tensor(lunarlander.action_space[2], dtype=torch.float32)

    net = neuralnet(state_size+action_size, action_size, [32])
    softmax_policy = Neural_SoftMax(net, actions)

    trajectories = lunarlander.simulate(10, 100, policy=softmax_policy)
   