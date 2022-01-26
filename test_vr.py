import torch
import os
import json
import torch.nn as nn
import argparse
from environments import MountainCar, Acrobot, LunarLander, CartPole
from methods import GPOMDP, A2C, REINFORCE, SVRPG_GPOMDP, SVRPG_REINFORCE, SRVRPG_GPOMDP, SRVRPG_REINFORCE, STORMPG_GPOMDP, STORMPG_REINFORCE
from policies import Neural_SoftMax, GaussianPolicy, neuralnet

parser = argparse.ArgumentParser(description='Training an RL Agent with REINFORCE-type Methods on Different Gym OpenAI Environments.')
parser.add_argument('-seed', type=int, default=0,
                    help='random seed')
parser.add_argument('-iterations', type=int, default=50,
                    help='number of iterations')
parser.add_argument('-method', type=str, default='SVRPG_GPOMDP',
                    help='method')
parser.add_argument('-env', type=str, default='Cartpole',
                    help='environment')
parser.add_argument('-alpha', type=float, default=0.001,
                    help='step-size')
parser.add_argument('-beta', type=float, default=0.001,
                    help='step-size')
parser.add_argument('-eta', type=float, default=0.001,
                    help='momentum-par')
parser.add_argument('-pt', type=float, default=0.4,
                    help='prob')
parser.add_argument('-batch_size', type=int, default=50,
                    help='batch_size')
parser.add_argument('-inner_batch_size', type=int, default=5,
                    help='inner_batch_size')
parser.add_argument('-inner_iterations', type=int, default=10,
                    help='inner_iterations')
parser.add_argument('-max_traj', type=int, default=3800,
                    help='bound on number of trajectories for PAGEPG')
args = parser.parse_args()

if args.method not in ['REINFORCE', 'GPOMDP', 'A2C', 'SVRPG_GPOMDP', 'SVRPG_REINFORCE', 'SRVRPG_GPOMDP', 'SRVRPG_REINFORCE', 'STORMPG_GPOMDP', 'STORMPG_REINFORCE', 'PAGEPG_GPOMDP', 'PAGEPG_REINFORCE']:
    raise Exception("The methods {} is not available.".format(args.method))


learning_rates = [args.alpha]
beta = args.beta

for alpha in learning_rates:

    if args.env == 'LunarLander':
        environment = LunarLander(render=True, seed=args.seed)
    elif args.env == 'MountainCar':
        environment = MountainCar(render=True, seed=args.seed)
        environment.env._max_episode_steps = 1000
    elif args.env == 'Acrobot':
        environment = Acrobot(render=True, seed=args.seed)
    elif args.env == 'Cartpole':
        environment = CartPole(render=True, seed=args.seed)
        environment.env._max_episode_steps = 200
    else:
        raise Exception("The selected environment is for now to available.")

    environment.set_seed()

    state_size = environment.state_space[1]
    action_size = environment.action_space[1]

    hidden_layers = [32, 32]
    net = neuralnet(state_size, action_size, hidden_layers, activation = nn.Tanh())
    
    softmax_policy = Neural_SoftMax(net, environment.action_space[2])

    iterations = args.iterations

    inner_iterations = args.inner_iterations

    if args.method in ['PAGEPG_GPOMDP','PAGEPG_REINFORCE', 'STORMPG_GPOMDP', 'STORMPG_REINFORCE']:
        iterations = 1
        inner_iterations = args.iterations
    
    batch_size = args.batch_size

    inner_batch_size = args.inner_batch_size

    avg_rwd = []
    tot_traj = []
    tot_policies = []

    discount_factor = 0.9999

    info = {"N": batch_size, "B": inner_batch_size, "iterations": iterations, "gamma": discount_factor, "alpha": alpha, "activation": "tanh", "hidden_layers": hidden_layers, "eta": args.eta}

   
    if args.method == "REINFORCE":
        method = REINFORCE(alpha, softmax_policy.neural_net.parameters, discount_factor)
    elif args.method == "GPOMDP":
        method = GPOMDP(alpha, softmax_policy.neural_net.parameters, discount_factor)
    elif args.method == "SVRPG_GPOMDP":
        method = SVRPG_GPOMDP(alpha, softmax_policy.neural_net.parameters, discount_factor)
    elif args.method == "SVRPG_REINFORCE":
        method = SVRPG_REINFORCE(alpha, softmax_policy.neural_net.parameters, discount_factor)
    elif args.method == "SRVRPG_GPOMDP":
        method = SRVRPG_GPOMDP(alpha, softmax_policy.neural_net.parameters, discount_factor)
    elif args.method == "SRVRPG_REINFORCE":
        method = SRVRPG_REINFORCE(alpha, softmax_policy.neural_net.parameters, discount_factor)
    elif args.method == "STORMPG_GPOMDP":
        method = STORMPG_GPOMDP(alpha, softmax_policy.neural_net.parameters, discount_factor, args.eta)
    elif args.method == "STORMPG_REINFORCE":
        method = STORMPG_REINFORCE(alpha, softmax_policy.neural_net.parameters, discount_factor, args.eta)
    elif args.method == "PAGEPG_GPOMDP":
        bernoulli = torch.distributions.bernoulli.Bernoulli(probs=args.pt)
        sequence = [bernoulli.sample() for i in range(10000)]
        #sequence1 = [bernoulli.sample() for i in range(10000)]
        #bernoulli = torch.distributions.bernoulli.Bernoulli(probs=0.6)
        #sequence2 = [bernoulli.sample() for i in range(10000)]
        #sequence = sequence1[:250] + sequence2[250:]
        seq = 0
        method = SRVRPG_GPOMDP(alpha, softmax_policy.neural_net.parameters, discount_factor)
    elif args.method == "PAGEPG_REINFORCE":
        bernoulli = torch.distributions.bernoulli.Bernoulli(probs=args.pt)
        sequence = [bernoulli.sample() for i in range(10000)]
        seq = 0
        method = SRVRPG_REINFORCE(alpha, softmax_policy.neural_net.parameters, discount_factor)
    else:
        V_function =  neuralnet(state_size, 1, [64, 64], activation = nn.Tanh())
        method = A2C(alpha, softmax_policy.neural_net.parameters, discount_factor, beta, V_function.parameters)

    tot_policies = [softmax_policy.neural_net.state_dict()]
    N_traj = 0 #batch_size 

    for iter in range(iterations):

        tot_reward = 0

        grad_full = []
        theta_snap = []

        trajectories = environment.simulate(batch_size, policy=softmax_policy)
        
        ##################################################
        if args.method == "A2C":
            loss, tot_reward, loss_q = method.loss(trajectories['states'], trajectories['actions'], trajectories['rewards'], softmax_policy, V_function)
        else:
            loss, tot_reward = method.loss(trajectories['states'], trajectories['actions'], trajectories['rewards'], softmax_policy)

        #compute gradient of the loss
        loss.backward()
        #step with the optimization method
        for par in softmax_policy.neural_net.parameters():
            grad_full.append(par.grad.clone())
            theta_snap.append(par.data.clone())

        method.step()
        N_traj += batch_size

        #saving the results
        avg_rwd.append(tot_reward/batch_size)
        tot_traj.append(N_traj)
        tot_policies.append(softmax_policy.neural_net.state_dict())

        print("Iteration [{}/{}], Trajectories [{}], avg. reward {}".format(iter+1, iterations, N_traj, avg_rwd[-1]))

        ##################################################
        if args.method == "A2C":
            
            loss_q.backward()
            method.critic_update()

            if N_traj >= args.max_traj:
                    break

        method.reset_grad()

        if args.method == 'SVRPG_GPOMDP' or args.method == 'SVRPG_REINFORCE':
            #inner loop 
            for inner_iter in range(inner_iterations):
                #resimulate
                trajectories = environment.simulate(inner_batch_size, policy=softmax_policy)
                #computed the corrected direction vt
                method.vt(trajectories['states'], trajectories['actions'], trajectories['rewards'], softmax_policy, theta_snap, grad_full)
                method.step()
                N_traj += inner_batch_size
                method.reset_grad()

            if N_traj >= args.max_traj:
                break

        elif args.method == 'SRVRPG_GPOMDP' or args.method == 'SRVRPG_REINFORCE':
            #refresh vold
            method.v_old = grad_full
            for inner_iter in range(inner_iterations):
                #resimulate
                trajectories = environment.simulate(inner_batch_size, policy=softmax_policy)
                #compute the corrected direction vt
                method.vt(trajectories['states'], trajectories['actions'], trajectories['rewards'], softmax_policy, theta_snap)
                method.step()
                N_traj += inner_batch_size
                method.reset_grad()

                #update theta_snap
                for ii, par in enumerate(softmax_policy.neural_net.parameters()):
                    theta_snap[ii] = par.data.clone()

            if N_traj >= args.max_traj:
                break

        elif args.method == "STORMPG_GPOMDP" or args.method == "STORMPG_REINFORCE":
            #refresh vold
            method.v_old = grad_full
            for inner_iter in range(inner_iterations):
                #resimulate
                trajectories = environment.simulate(inner_batch_size, policy=softmax_policy)
                #compute the corrected direction vt
                method.vt(trajectories['states'], trajectories['actions'], trajectories['rewards'], softmax_policy, theta_snap)
                method.step()
                N_traj += inner_batch_size
                method.reset_grad()

                tot_reward = 0
                    
                for ii in range(inner_batch_size): 

                    rewards = torch.tensor(trajectories["rewards"][ii])
                    tot_reward += rewards.sum()

                avg_rwd.append(tot_reward.item()/inner_batch_size)
                tot_traj.append(N_traj)
                tot_policies.append(softmax_policy.neural_net.state_dict())

                #update theta_snap
                for ii, par in enumerate(softmax_policy.neural_net.parameters()):
                    theta_snap[ii] = par.data.clone()

                print("Iteration [{}/{}], Trajectories [{}], avg. reward {}".format(inner_iter+1, inner_iterations, N_traj, avg_rwd[-1]))

                if N_traj >= args.max_traj:
                    break

        elif args.method == 'PAGEPG_GPOMDP' or args.method == 'PAGEPG_REINFORCE' or args.method == 'MOMENTUMPAGEPG_GPOMDP' or args.method == 'MOMENTUMPAGEPG_REINFORCE':
            
            method.v_old = grad_full
            for inner_iter in range(inner_iterations):
                
                if sequence[seq]:
                    #resimulate
                    trajectories = environment.simulate(inner_batch_size, policy=softmax_policy)
                    method.vt(trajectories['states'], trajectories['actions'], trajectories['rewards'], softmax_policy, theta_snap)
                    N_traj += inner_batch_size
                    
                    tot_reward = 0
                    for ii in range(inner_batch_size): 

                        rewards = torch.tensor(trajectories["rewards"][ii])
                        tot_reward += rewards.sum()

                    avg_rwd.append(tot_reward.item()/inner_batch_size)
                    tot_traj.append(N_traj)

                    #update theta_snap
                    print("Short Iteration [{}/{}], Trajectories [{}], avg. reward {}".format(iter+1, inner_iterations, N_traj, avg_rwd[-1]))

                else:
                    #resimulate
                    trajectories = environment.simulate(batch_size, policy=softmax_policy)
                    loss, tot_reward = method.loss(trajectories['states'], trajectories['actions'], trajectories['rewards'], softmax_policy)
                    loss.backward() 
                    grad_full = []
                    for par in softmax_policy.neural_net.parameters():
                        grad_full.append(par.grad.clone())
                    method.v_old = grad_full
                    N_traj += batch_size
                    avg_rwd.append(tot_reward/batch_size)
                    tot_traj.append(N_traj)
      
                    tot_policies.append(softmax_policy.neural_net.state_dict())

                    print("Iteration [{}/{}], Trajectories [{}], avg. reward {}".format(iter+1, inner_iterations, N_traj, avg_rwd[-1]))

                method.step()

                method.reset_grad()

                for ii, par in enumerate(softmax_policy.neural_net.parameters()):
                    theta_snap[ii] = par.data.clone()
                
                iter += 1
                seq += 1
               
                if N_traj >= args.max_traj:
                    break
        
    res = {"avg_rwd": avg_rwd, "trajectories": tot_traj, "info": info}

    directory = "results_lr_{}_svrpg/{}".format(alpha, args.env)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, "{}_res_run_{}.json".format(args.method, args.seed)), 'w') as f:
        json.dump(res, f)

    torch.save(tot_policies, os.path.join(directory, "{}_policies_run_{}.json".format(args.method, args.seed)))



