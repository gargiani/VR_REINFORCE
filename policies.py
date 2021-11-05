import torch
import torch.nn as nn
import numpy as np


class GaussianPolicy:

    def __init__(self, neural_mean, variance):

        self.variance = variance
        self.mean = neural_mean
        self.policy = None
        
    def distribution(self, observation):
        
        self.policy = torch.distributions.multivariate_normal.MultivariateNormal(self.mean(observation), covariance_matrix=self.variance)

    def sample(self):

        if self.policy == None:
            raise ValueError("Distribution not defined!")
        else:
            return self.policy.sample()     

    def log_prob(self, observations, actions, batch_size):

        self.distribution(observations)
        
        return (1/batch_size)*self.policy.log_prob(actions) 
        
class Neural_SoftMax:

    def __init__(self, neural_net, action_space):

        self.actions = action_space
        self.neural_net = neural_net
        self.softmax = nn.Softmax(dim=1)
        self.policy = None 

    def distribution(self, observations):

        input = torch.cat((observations, self.actions.repeat(observations.shape[0], 1)), dim=1)
        features = self.neural_net(input)
        self.policy = self.softmax(features)

    def sample(self):
        
        if self.policy == None:
            raise ValueError("Distribution not defined!")
        else:
            return torch.argmax(self.policy, dim=1)

    def log_prob(self, observations, actions, batch_size):

        self.distribution(observations)
        actions_idx = torch.unsqueeze(actions, 1)
        
        return (1/batch_size)*torch.log(torch.gather(self.policy, 1, actions_idx))
        
        
class neuralnet(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, activation = nn.Tanh()):
        super(neuralnet, self).__init__()

        modules = []
        hidden_layers.insert(0, input_size)
        
        for ii, _ in enumerate(hidden_layers[:-1]):
            modules.append(nn.Linear(hidden_layers[ii], hidden_layers[ii+1]))
            modules.append(activation)
        
        modules.append(nn.Linear(hidden_layers[-1], output_size))

        self.sequential = nn.Sequential(*modules)

    def forward(self, x):    
        return self.sequential(x)

if __name__ == "__main__":
    
    ##Gaussian Policy with Neural Mean
    state_size = 48
    action_size = 2
    #create container for Gaussian policy
    net = neuralnet(state_size, action_size, [45, 65])
    gaussian_policy = GaussianPolicy(net, torch.eye(action_size))
    #create the distribution based on the observations
    gaussian_policy.distribution(torch.ones((23,state_size)))
    #sample from the distribution
    gaussian_policy.sample()
    
    ##Neural SoftMax Policy
    state_size = 5
    action_size = 3
    N_samples = 10
    observations = torch.rand(N_samples, state_size)
    actions = torch.rand(3)

    #create container for softmax policy
    net = neuralnet(state_size+action_size, action_size, [23, 45])
    softmax_policy = Neural_SoftMax(net, actions)
    #create the distribution
    softmax_policy.distribution(observations)
    #sample from the distribution
    softmax_policy.sample()


    import pdb; pdb.set_trace()
    print("end test")