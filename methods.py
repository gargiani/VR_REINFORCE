import torch
from torch.autograd import Variable
import numpy as np

class NumericalMethod:

    def __init__(self, alpha, parameters, discount_factor):

        self.alpha = alpha
        self.discount_factor = discount_factor
        self.parameters = parameters

    def R_tau(self, rewards):
        
        return torch.tensor([self.discount_factor**i*rewards[i] for i in range(rewards.shape[0])]).sum().item()
        
    def R_togo(self, rewards):

        Rewards = rewards.repeat((rewards.shape[0], 1))
        gamma = torch.tensor([self.discount_factor**i for i in range(rewards.shape[0])])
        Gamma = torch.triu(gamma.repeat((gamma.shape[0], 1)))

        return (Gamma*Rewards).sum(dim=1)

    def importance_weights(self, states, actions, theta_new, theta_snap, policy):
        
        p_new = [1]*len(states)
        p_old = [1]*len(states)

        for ii, par in enumerate(policy.neural_net.parameters()):
            par.data = theta_new[ii]

        for ii, states_i in enumerate(states):
            actions_i = actions[ii]
            policy.distribution(torch.tensor(states_i))
            distr_ = policy.policy
            for jj, action in enumerate(actions_i):
                p_new[ii] = p_new[ii]*distr_[jj, action]

        for ii, par in enumerate(policy.neural_net.parameters()):
            par.data = theta_snap[ii]

        for ii, states_i in enumerate(states):
            actions_i = actions[ii]
            policy.distribution(torch.tensor(states_i))
            distr_ = policy.policy
            for jj, action in enumerate(actions_i):
                p_old[ii] = p_old[ii]*distr_[jj, action]
        
        return  torch.tensor([(p_old[jj]+0*self.TINY)/(p_new[jj]+self.TINY) for jj, _ in enumerate(p_new)]).detach()


    def step(self):
        
        for ii, par in enumerate(self.parameters()):
            par.data = par.data + self.alpha*par.grad
          
    def reset_grad(self):

        for par in self.parameters():
            par.grad.data.zero_()

        self.reset_vgrad()

    def reset_vgrad(self):

        pass

class REINFORCE(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)

    def grad_estimator(self, log_prob, rewards):

        estimator = self.R_tau(rewards)*log_prob.sum()
        estimator.backward(retain_graph=True)

    def loss(self, states, actions, rewards, policy):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0
        
        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1).sum(dim=0)) * self.R_tau(r_n)
            tot_rwd += r_n.sum().item()
            
        return logprob.mean(), tot_rwd
        

class GPOMDP(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
    
    def grad_estimator(self, log_prob, rewards):

        estimator = (self.R_togo(rewards)*log_prob).sum()
        estimator.backward(retain_graph=True)

    def loss(self, states, actions, rewards, policy):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0

        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1)*self.R_togo(r_n)).sum(dim=0)
            tot_rwd += r_n.sum().item()
            
        return logprob.mean(), tot_rwd

class A2C(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor, beta, v_parameters):
        super(A2C, self).__init__(alpha, parameters, discount_factor)
        self.beta = beta
        self.v_parameters = v_parameters

    def loss(self, states, actions, rewards, policy, value_function):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        loss_q=torch.empty(N, requires_grad=False)
        tot_rwd = 0

        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])

            gamma = torch.tensor([self.discount_factor**i for i in range(s_n.shape[0])])

            output = value_function(s_n).squeeze(-1).detach()
            advantage_fun = r_n + self.discount_factor*torch.cat((output[1:], torch.zeros(1)), 0) - output
            
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1)*(gamma*advantage_fun)).sum(dim=0)
            #using Monte Carlo
            loss_q[n] = ((r_n[:-1] + self.discount_factor*value_function(s_n[1:]).squeeze(-1).detach() - value_function(s_n[:-1]).squeeze(-1))**2).sum()

            tot_rwd += r_n.sum().item()
            
        return logprob.mean(), tot_rwd, loss_q.sum()
    
    def reset_vgrad(self):

        for par in self.v_parameters():
            par.grad.data.zero_()

    def critic_update(self):

        for ii, par in enumerate(self.v_parameters()):
            par.data = par.data - self.beta*par.grad
        

class SVRPG_GPOMDP(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
        self.TINY = 10**-9

    def loss(self, states, actions, rewards, policy, weights=None):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0

        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1)*self.R_togo(r_n)).sum(dim=0)
            tot_rwd += r_n.sum().item()

        if weights is not None:
            logprob = logprob*weights

        return logprob.mean(), tot_rwd
    
    def vt(self, states, actions, rewards, policy, theta_snap, grad_full):
        
        grad_new = []
        grad_old = []
        theta_new = []

        #compute grad of current iterate for current data
        loss, _ = self.loss(states, actions, rewards, policy)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_new.append(par.grad.clone())
            theta_new.append(par.data)
            par.data = theta_snap[ii]
        
        self.reset_grad()

        omega = self.importance_weights(states, actions, theta_new, theta_snap, policy)

        for ii, par in enumerate(self.parameters()):
            par.data = theta_snap[ii]
        
        #compute grad of current iterate for past data
        loss, _ = self.loss(states, actions, rewards, policy, weights=omega)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_old.append(par.grad.clone())
            par.data = theta_new[ii]

        self.reset_grad()

        for ii, par in enumerate(self.parameters()):
            
            par.grad = grad_full[ii] + grad_new[ii] - grad_old[ii]
            
           
class SVRPG_REINFORCE(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
        self.TINY = 10**-9

    def loss(self, states, actions, rewards, policy, weights=None):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0
        
        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1).sum(dim=0)) * self.R_tau(r_n)
            tot_rwd += r_n.sum().item()

        if weights is not None:
            logprob = logprob*weights
            
        return logprob.mean(), tot_rwd
        
    
    def vt(self, states, actions, rewards, policy, theta_snap, grad_full):
        
        grad_new = []
        grad_old = []
        theta_new = []
        #compute grad of current iterate for current data
        loss, _ = self.loss(states, actions, rewards, policy)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_new.append(par.grad.clone())
            theta_new.append(par.data)
            par.data = theta_snap[ii]

        self.reset_grad()

        omega = self.importance_weights(states, actions, theta_new, theta_snap, policy)

        #compute grad of current iterate for past data
        loss, _ = self.loss(states, actions, rewards, policy, weights=omega)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_old.append(par.grad.clone())
            par.data = theta_new[ii]

        self.reset_grad()


        for ii, par in enumerate(self.parameters()):
            par.grad = grad_full[ii] + grad_new[ii] - grad_old[ii]
    
  
class SRVRPG_GPOMDP(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
        self.TINY = 10**-9
        self.v_old = None

    def loss(self, states, actions, rewards, policy, weights=None):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0

        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1)*self.R_togo(r_n)).sum(dim=0)
            tot_rwd += r_n.sum().item()
        if weights is not None:
            logprob = logprob*weights

        return logprob.mean(), tot_rwd
    
    def vt(self, states, actions, rewards, policy, theta_snap):
        
        grad_new = []
        grad_old = []
        theta_new = []
        #compute grad of current iterate for current data
        loss, _ = self.loss(states, actions, rewards, policy)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_new.append(par.grad.clone())
            theta_new.append(par.data)
            par.data = theta_snap[ii]

        self.reset_grad()

        omega = self.importance_weights(states, actions, theta_new, theta_snap, policy)


        for ii, par in enumerate(self.parameters()):
            par.data = theta_snap[ii]

        #compute grad of current iterate for past data
        loss, _ = self.loss(states, actions, rewards, policy, weights=omega)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_old.append(par.grad.clone())
            par.data = theta_new[ii]

        self.reset_grad()

        for ii, par in enumerate(self.parameters()):
            par.grad = self.v_old[ii] + grad_new[ii] - grad_old[ii]
            self.v_old[ii] = self.v_old[ii] + grad_new[ii] - grad_old[ii]

class SRVRPG_REINFORCE(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor):
        super().__init__(alpha, parameters, discount_factor)
        self.TINY = 10**-6
        self.v_old = None

    def loss(self, states, actions, rewards, policy):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0
        
        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1).sum(dim=0)) * self.R_tau(r_n)
            tot_rwd += r_n.sum().item()
            
        return logprob.mean(), tot_rwd
    
    def vt(self, states, actions, rewards, policy, theta_snap):
        
        grad_new = []
        grad_old = []
        theta_new = []
        #compute grad of current iterate for current data
        loss, _ = self.loss(states, actions, rewards, policy)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_new.append(par.grad.clone())
            theta_new.append(par.data)
            par.data = theta_snap[ii]

        self.reset_grad()

        omega = self.importance_weights(states, actions, theta_new, theta_snap, policy)

        #compute grad of current iterate for past data
        loss, _ = self.loss(states, actions, rewards, policy, weights=omega)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_old.append(par.grad.clone())
            par.data = theta_new[ii]

        self.reset_grad()


        for ii, par in enumerate(self.parameters()):
            par.grad = self.v_old[ii] + grad_new[ii] - grad_old[ii]
            self.v_old[ii] = self.v_old[ii] + grad_new[ii] - grad_old[ii]



class STORMPG_GPOMDP(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor, eta):
        super().__init__(alpha, parameters, discount_factor)
        self.TINY = 10**-6
        self.v_old = None
        self.eta = eta

    def loss(self, states, actions, rewards, policy, weights=None):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0

        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1)*self.R_togo(r_n)).sum(dim=0)
            tot_rwd += r_n.sum().item()
        if weights is not None:
            logprob = logprob*weights

        return logprob.mean(), tot_rwd
    
    def vt(self, states, actions, rewards, policy, theta_snap):
        
        grad_new = []
        grad_old = []
        theta_new = []
        #compute grad of current iterate for current data
        loss, _ = self.loss(states, actions, rewards, policy)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            #print(par.grad)
            grad_new.append(par.grad.clone())
            theta_new.append(par.data)
            par.data = theta_snap[ii]

        self.reset_grad()

        omega = self.importance_weights(states, actions, theta_new, theta_snap, policy)

        #compute grad of current iterate for past data
        loss, _ = self.loss(states, actions, rewards, policy, weights=omega)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_old.append(par.grad.clone())
            par.data = theta_new[ii]

        self.reset_grad()


        for ii, par in enumerate(self.parameters()):
            
            par.grad = (1-self.eta)*(self.v_old[ii] + grad_new[ii] - grad_old[ii]) + self.eta*grad_new[ii]
            self.v_old[ii] = (1-self.eta)*(self.v_old[ii] + grad_new[ii] - grad_old[ii]) + self.eta*grad_new[ii]

class STORMPG_REINFORCE(NumericalMethod):

    def __init__(self, alpha, parameters, discount_factor, eta):
        super().__init__(alpha, parameters, discount_factor)
        self.TINY = 10**-6
        self.v_old = None
        self.eta = eta

    def loss(self, states, actions, rewards, policy, weights=None):

        #extract number of trajectories
        N = len(states)
        logprob=torch.empty(N, requires_grad=False)
        tot_rwd = 0
        
        for n in range(N):
            s_n = torch.tensor(states[n])
            a_n = torch.tensor(actions[n])
            r_n = torch.tensor(rewards[n])
            #compute the logprob
            logprob[n] = (policy.log_prob(s_n, a_n , 1).squeeze(-1).sum(dim=0)) * self.R_tau(r_n)
            tot_rwd += r_n.sum().item()

        if weights is not None:
            logprob = logprob*weights
            
        return logprob.mean(), tot_rwd

    def vt(self, states, actions, rewards, policy, theta_snap):
        
        grad_new = []
        grad_old = []
        theta_new = []
        #compute grad of current iterate for current data
        loss, _ = self.loss(states, actions, rewards, policy)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_new.append(par.grad.clone())
            theta_new.append(par.data)
            par.data = theta_snap[ii]

        self.reset_grad()

        omega = self.importance_weights(states, actions, theta_new, theta_snap, policy)

        #compute grad of current iterate for past data
        loss, _ = self.loss(states, actions, rewards, policy, weights=omega)
        loss.backward()

        for ii, par in enumerate(self.parameters()):
            grad_old.append(par.grad.clone())
            par.data = theta_new[ii]

        self.reset_grad()


        for ii, par in enumerate(self.parameters()):
            par.grad = (1-self.eta)*(self.v_old[ii] + grad_new[ii] - grad_old[ii]) + self.eta*grad_new[ii]
            self.v_old[ii] = (1-self.eta)*(self.v_old[ii] + grad_new[ii] - grad_old[ii]) + self.eta*grad_new[ii]