import os
import argparse
import time
import numpy as np

from datetime import datetime
from copy import deepcopy

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from model.KellerMiksis import Keller
from model.NeuralODE import NeuralODE

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--name',
    default = '',
    type    = str,
    help    = "Optional name of the model."
)
parser.add_argument(
    '--method',
    type=str, 
    choices=['dopri5', 'adams'], 
    default='dopri5'
)
parser.add_argument(
    '--data_size',
    type=int, 
    default=1000
)
parser.add_argument(
    '--batch_time', 
    type=int, 
    default=10,
    help="Number of time data to be used as one training point. "
)#see get_batch() function to fully understand its role
parser.add_argument(
    '--batch_size', 
    type=int, 
    default=20
)
parser.add_argument(
    '--lr',
    type=float,
    default=0.01
)
parser.add_argument(
    '--iters', 
    type=int, 
    default=2000
)
parser.add_argument(
    '--print_iter', 
    type=int, 
    default=20
)
parser.add_argument(
    '--viz', 
    action='store_true'
)
parser.add_argument(
    '--gpu', 
    type=int, 
    default=0,
    help="GPU device id to use for pytorch computations. Default is 0."
)
parser.add_argument(
    '--early_stop', 
    type=int, 
    default=0,
    help="Stop early when no inprovement in loss was detected for the given amount of iterations. Default is 0 meaning early stop turned off."
)
parser.add_argument(
    '--adjoint', 
    action='store_true'
    )
parser.add_argument(
    '--num_threads',
    default = 0,
    type    = int,
    help    = "Number of cpu threads to be used by pytorch. Default is 0 meaning same as number of cores."
)
parser.add_argument(
    '--cpu',
    dest='cpu', 
    action='store_true', 
    help= "If set, training is carried out on the cpu."
)
parser.add_argument(
    '--save_path',
    default = 'training/',
    type    = str,
    help    = "Path to save model. Default is 'training'."
)
parser.set_defaults(
    cpu=False
)
args = parser.parse_args()

#set number of threads used by pytorch
if args.num_threads:
    torch.set_num_threads(args.num_threads)
    
#make savedir dierctory
if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)

#use adjoint version or automatic differentiation
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

#select device
#device selection logic
device=0
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

start_iter = 1

#beginning initializations
begin_time = datetime.now();
time_str = begin_time.strftime("%y%m%d%H%M")
print("Begin: "+ str(time_str))
logfile = open(args.save_path + (args.name+'_' if args.name else '') + time_str + '.log','w')


#initial condition
true_y0 = torch.tensor([[1., 0.]]).to(device)
#
t = torch.linspace(0., 2.5, args.data_size).to(device)


#solving the original ODE
with torch.no_grad():
    original_true_y = odeint(Keller(), true_y0, t, method='dopri5', rtol=1e-12, atol=1e-12)

print(original_true_y[:,0,:].shape)
scaler  = MinMaxScaler(feature_range=(0, 1), copy=False)
#QuantileTransformer(output_distribution="normal", random_state=42, copy=False)#StandardScaler(with_mean=True, with_std=True, copy=False)
scaler.fit(original_true_y.detach().numpy()[:,0,:])
true_y   = torch.tensor(scaler.transform(original_true_y.detach().numpy()[:,0,:]).reshape((args.data_size,1,2)))
print(true_y.shape)
#
def get_batch():
    #random data indices
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False)) 
    #exact values -> initial conditions
    batch_y0 = true_y[s]  # (M, D)
    #time points
    batch_t = t[:args.batch_time]  # (T)
    #all together
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    #plt.plot(t, true_y[:,0,0])
    #plt.plot(t, true_y[:,0,1])
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)



def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-5, 5)
        ax_phase.set_ylim(-5, 5)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-5:5:21j, -5:5:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-5, 5)
        ax_vecfield.set_ylim(-5, 5)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    

    model = NeuralODE(2,2).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    loss_arr = np.zeros(args.iters)
    iterations = np.linspace(start_iter, start_iter+args.iters, args.iters)

    
    best_model_state_dict = deepcopy(model.state_dict())
    best_optim_state_dict = deepcopy(optimizer.state_dict())
    best_loss = 1e100
    best_iter = 0
    
    print("Training...",file=logfile)

    for itr in range(1, args.iters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(model, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        
        
        with torch.no_grad():
            pred_y = odeint(model, true_y0, t)
            loss = torch.mean(torch.abs(pred_y - true_y)).item()
            loss_arr[itr-1] = loss
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss), file=logfile)
            
            if loss < best_loss:
                best_loss = loss
                best_model_state_dict = deepcopy(model.state_dict())
                best_optim_state_dict = deepcopy(optimizer.state_dict())
                best_iter = itr
                if args.early_stop and itr-best_iter==args.early_stop:
                    print("Early stopped", file=logfile)
                    print("Early stopped")
                    break
                    
            
            if itr % args.print_iter == 0:
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss))
                visualize(true_y, pred_y, model, itr)


    end_time = datetime.now()
    duration = end_time - begin_time
    time_end_str = end_time.strftime("%y%m%d%H%M")

    print("Training ready", file=logfile)
    print("Ended at: "+ time_end_str,  file=logfile)
    print("Duration: " + str(duration), file=logfile)
    print("Ended at: "+ time_end_str)
    print("Duration: " + str(duration))


    # ----- ----- ----- ----- ----- -----
    #Model Save
    # ----- ----- ----- ----- ----- -----

    traced_model = 0

    torch.save({
        'iteration': start_iter+best_iter,
        'model_state_dict': best_model_state_dict,
        #'scheduler_state_dict': best_scheduler_state_dict,
        'optimizer_state_dict': best_optim_state_dict
        },
        args.save_path+'model_' + (args.name+'_' if args.name else '') + 'i' + str(start_iter+best_iter) + '_' + time_str + '.pt')
    if not args.early_stop:
        torch.save({
            'iteration': start_iter+args.iters,
            'model_state_dict': model.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            },
            args.save_path+'model_' + (args.name+'_' if args.name else '') + 'i' + str(start_iter+args.iters) + '_' + time_str + '.pt')
    print("Saved model.",file=logfile)
    print("Saved model.")
    
    import pickle
    pickle_path = args.save_path+'scaler_' + (args.name+'_' if args.name else '') +time_str + '.psca.pickle'
    with open(pickle_path, "wb") as file:
        pickle.dump(scaler, file)
    print("Saved scaler.",file=logfile)
    print("Saved scaler.")

    #trace model to be used by C/C++
    model.load_state_dict(best_model_state_dict)
    model.eval()
    traced_model = torch.jit.trace(model.cpu(), (t[0], true_y0))
    traced_model_path = args.save_path+'traced_model_' + (args.name+'_' if args.name else '') + 'i'+str(start_iter+best_iter) + '_' + time_str + '.pt'
    traced_model.save(traced_model_path)
    print("Saved trace model.",file=logfile)
    print("Saved trace model.")

    # ----- ----- ----- ----- ----- -----
    # Plotting
    # ----- ----- ----- ----- ----- -----
     
    if args.viz:
        plt.ion()
        plt.show()
    plt.plot(iterations, loss_arr, label='Total Loss')
    plt.yscale('log')
    plt.title('Loss Diagram')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    if args.viz:
        plt.show()
    
    plt.savefig(args.save_path+"learning_curve_"+ (args.name+'_' if args.name else '') + time_str+".pdf")

    if args.viz:
        plt.ioff()
        plt.show()
    
logfile.close()