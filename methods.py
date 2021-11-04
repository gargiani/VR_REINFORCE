import torch

class NumericalMethod:

    def __init__(self, alpha, parameters, discount_factor):

        self.alpha = alpha
        self.discount_factor = discount_factor
        self.parameters = parameters

    def R_tau(self, rewards):
        
        return (torch.tensor([self.discount_factor**i for i in range(rewards.shape[0])])*rewards).sum().item()
        
    def R_togo(self, rewards):
        #reward to go
        print("TO DO!")
    
    def step(self):
        
        for par in self.parameters():
            par = par + self.alpha*par.grad    

    def reset_grad(self):

        for par in self.parameters():
            par.grad.data.zero_()

class REINFORCE(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)

    def grad_estimator(self, avg_score, rewards):

        estimator = self.R_tau(rewards)*avg_score
        estimator.backward(retain_graph=True)




class GPOMDP(NumericalMethod):

    def __init__(self, alpha, discount_factor):
        super().__init__(alpha, discount_factor)
        print("TO DO!")

class SVRPG(NumericalMethod):

    def __init__(self, alpha, discount_factor):
        super().__init__(alpha, discount_factor)
        print("TO DO!")

class SRVRPG(NumericalMethod):

    def __init__(self, alpha, discount_factor):
        super().__init__(alpha, discount_factor)
        print("TO DO!")

class STORMPG(NumericalMethod):

    def __init__(self, alpha, discount_factor):
        super().__init__(alpha, discount_factor)
        print("TO DO!")


