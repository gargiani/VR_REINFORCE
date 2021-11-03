from policies import GaussianPolicy
import torch
import gym
import numpy as np

class Environment:

    def __init__(self, render=False):
 
       self.render = render

    def simulate(self, N, T, policy=None):

        states_n = []
        actions_n = []
        rewards_n = []

        for episode in range(N):

            states = []
            actions = []
            rewards = []

            observation = self.env.reset()
            states.append(observation.tolist())
            
            for step in range(T):
                
                if self.render:
                    self.env.render()
                
                if policy==None:
                    action = self.env.action_space.sample()
                    print(action)
                else:
                    policy.distribution(torch.tensor([observation], dtype=torch.float32))
                    action = policy.sample()[0].numpy()#.item()

                observation, reward, done, info = self.env.step(action)
                
                states.append(observation.tolist())
                rewards.append(reward)
                actions.append(action)

                if done: 
                    print("done step {}".format(step))
                    observation = self.env.reset()

            states_n.append(states)
            actions_n.append(actions)
            rewards_n.append(rewards)

        self.env.close()
        
        return {"states": states_n, "actions": actions_n, "rewards": rewards_n}


#CartPole
class CartPole(Environment):

    def __init__(self, render=False):
        super().__init__(render)

        self.env = gym.make('CartPole-v1')
        self.state_space = ("Continuous", 4)
        self.action_space = ("Discrete", 2, [0,1])

#Pendulum
class Pendulum(Environment):

    def __init__(self, render=False):
        super().__init__(render)

        self.env = gym.make('Pendulum-v0')
        self.state_space = ("Continuous", 3)
        self.action_space = ("Continuous", 1)


#MountainCar
class MountainCar(Environment):

    def __init__(self, render=False):
        super().__init__(render)

        self.env = gym.make('MountainCarContinuous-v0')
        self.state_space = ("Continuous", 2)
        self.action_space = ("Continuous", 1)

#BipedalWalker
class BipedalWalker(Environment):

    def __init__(self, render=False):
        super().__init__(render)

        self.env = gym.make('BipedalWalker-v3')
        self.state_space = ("Continuous", 24)
        self.action_space = ("Continuous", 4)

#LunarLanderContinuous
class LunarLanderContinuous(Environment):

    def __init__(self, render=False):
        super().__init__(render)

        self.env = gym.make('LunarLanderContinuous-v2')
        self.state_space = ("Continuous", 8)
        self.action_space = ("Continuous", 2)

#LunarLander
class LunarLander(Environment):

    def __init__(self, render=False):
        super().__init__(render)

        self.env = gym.make('LunarLander-v2')
        self.state_space = ("Continuous", 8)
        self.action_space = ("Discrete", 4, [0, 1, 2, 3])


if __name__ == "__main__":

    
    from policies import Neural_SoftMax, neuralnet
    
    #CARTPOLE SCENARIO
    cartpole = CartPole(render=True)

    state_size = cartpole.state_space[1]
    action_size = cartpole.action_space[1]
    
    actions = torch.tensor(cartpole.action_space[2], dtype=torch.float32)

    net = neuralnet(state_size+action_size, action_size, [32])
    softmax_policy = Neural_SoftMax(net, actions)
    
    trajectories = cartpole.simulate(10, 100, policy=softmax_policy)
   
    #PENDULUM
    pendulum = Pendulum(render=True)

    state_size = pendulum.state_space[1]
    action_size = pendulum.action_space[1]
    
    net = neuralnet(state_size, action_size, [32])
    gaussian_policy = GaussianPolicy(net, torch.eye(action_size))
    
    trajectories = pendulum.simulate(10, 100, policy=gaussian_policy)

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
   