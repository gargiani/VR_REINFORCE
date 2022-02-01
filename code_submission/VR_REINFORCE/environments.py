import torch
import gym
import random
import numpy as np

torch.backends.cudnn.deterministic=True

class Environment:

    def __init__(self, render=False, seed=None):
 
       self.render = render
       self.env_seed = seed

    def set_seed(self):

        if self.env_seed is not None:
            self.env.seed(self.env_seed)   
            self.env.action_space.seed(self.env_seed)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.random.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def simulate(self, N, policy=None, verbose=False):

        states_n = []
        actions_n = []
        rewards_n = []

        tot_reward = 0

        for episode in range(N):
            
            if verbose:
                print("episode {} of {}\n".format(episode+1, N))

            states = []
            actions = []
            rewards = []

            done = False

            observation = self.env.reset()

            while not done:
                
                if self.render:
                    self.env.render()   

                states.append(observation.tolist()) 
                
                if policy==None:                    
                    action = self.env.action_space.sample()
                    action = np.asarray(action)
                else:
                    policy.distribution(torch.tensor([observation], dtype=torch.float32))
                    action = policy.sample()[0].numpy()
                observation, reward, done, _ = self.env.step(action)
                
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
class MountainCarContinous(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('MountainCarContinuous-v0')
        self.state_space = ("Continuous", 2)
        self.action_space = ("Continuous", 1)

#MountainCar-Discrete
class MountainCar(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('MountainCar-v0')
        self.state_space = ("Continuous", 2)
        self.action_space = ("Discrete", 3, [0,1,2])

#Acrobot-Discrete
class Acrobot(Environment):

    def __init__(self, render=False, seed=None):
        super().__init__(render, seed)

        self.env = gym.make('Acrobot-v1')
        self.state_space = ("Continuous", 6)
        self.action_space = ("Discrete", 3, [0,1,2])

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

    #cartpole
    cartpole = CartPole(render=True)
    cartpole.simulate(6)

    #pendulum
    pendulum = Pendulum(render=True)
    pendulum.flag_terminal_reward = True
    pendulum.simulate(3)

    #mountaincar
    mountaincar = MountainCar(render=True)
    mountaincar.simulate(3)

    #acrobot
    acrobot = Acrobot(render=True)
    acrobot.simulate(3)

    #bipedalwalker
    bipedalwalker = BipedalWalker(render=True)
    bipedalwalker.simulate(3)

    #lunarlander
    lunarlander = LunarLander(render=True)
    lunarlander.simulate(3)