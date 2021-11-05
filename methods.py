import torch

class NumericalMethod:

    def __init__(self, alpha, parameters, discount_factor):

        self.alpha = alpha
        self.discount_factor = discount_factor
        self.parameters = parameters

    def R_tau(self, rewards):
        
        return (torch.tensor([self.discount_factor**i for i in range(rewards.shape[0])])*rewards).sum().item()
        
    def R_togo(self, rewards):

        Rewards = rewards.repeat((rewards.shape[0], 1))
        gamma = torch.tensor([self.discount_factor**i for i in range(rewards.shape[0])])
        Gamma = torch.triu(gamma.repeat((gamma.shape[0], 1)))

        return (Gamma*Rewards).sum(dim=1)

    def step(self):
        
        for par in self.parameters():
            par = par + self.alpha*par.grad    

    def reset_grad(self):

        for par in self.parameters():
            par.grad.data.zero_()

class REINFORCE(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)

    def grad_estimator(self, log_prob, rewards):

        estimator = self.R_tau(rewards)*log_prob.sum()
        estimator.backward(retain_graph=True)

class GPOMDP(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
    
    def grad_estimator(self, log_prob, rewards):

        estimator = (self.R_togo(rewards)*log_prob).sum()
        estimator.backward(retain_graph=True)

class SVRPG(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
        print("TO DO!")

class SRVRPG(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
        print("TO DO!")

class STORMPG(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
        print("TO DO!")


