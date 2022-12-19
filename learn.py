
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import torch
import inverse

def train_motors(For_model, model):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for n in range(100):
        inputs, target = For_model.generate_maps(5000)
        train_loop(model, inputs, target, distance_loss, optimizer)
        #if n%10==0: optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.8    

def train_loop(model, inputs, target, loss_fn, n_reps = 10, optimizer=None):
    
    if optimizer is None:  optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sumloss = 0 
    model.train()
    n_reps = 10
    for n in range(n_reps):
        n = np.random.randint(len(inputs),size=5000)
        X = inputs[n]
        Y = target[n]
        tX = torch.Tensor(X)
        # Compute prediction and loss
        pred = model(tX)
        tY = torch.Tensor(Y)
        loss = loss_fn(pred, tY)
        # Backpropagation
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        sumloss = sumloss+loss.mean().item()

    print(sumloss/n_reps)
    
def train_positions(For_model, model, n_cycles = 500):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    For_model
    
    for n in range(n_cycles):
        inputs, target = For_model.generate_maps(1000)
        inputs = torch.Tensor(inputs)

        pred = model(inputs)

        if inputs.shape[-1]==3:         orientation=False
        else:                           orientation=True
     
        pred_xys = inverse.with_torch().forward_from_active(For_model, pred, orientation=orientation)
        pred_xys = pred_xys[:,-1,:]

        loss = distance_loss(pred_xys, inputs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(np.array(loss.item()).round(2))       

        if n%100==0: optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.5  
        #if n%100==0: optimizer.param_groups[0]["lr"] = 1e-3  

def evaluate_plot(For_model, model):
    
    inputs, target = For_model.generate_maps(1000)
    inputs = torch.Tensor(inputs)
    model.eval()

    pred = model(inputs)

    if inputs.shape[-1]==3:         orientation=False
    else:                           orientation=True
     
    pred_xys = inverse.with_torch().forward_from_active(For_model, pred, orientation=orientation)

    loss = distance_loss(pred_xys, inputs)
    print(loss)

    pred_xys = np.array(pred_xys.detach()).squeeze()
    inputs = np.array(inputs)

    fig,axs = plt.subplots(1,2)
    axs[0].plot(pred_xys[:,0],pred_xys[:,1],"ro")
    axs[0].plot(inputs[:,0],inputs[:,1],"bo")
    axs[0].plot([pred_xys[:,0],inputs[:,0]], [pred_xys[:,1],inputs[:,1]],"g-")

    axs[1].plot(pred_xys[:,0],pred_xys[:,2],"ro")
    axs[1].plot(inputs[:,0],inputs[:,2],"bo")
    axs[1].plot([pred_xys[:,0],inputs[:,0]], [pred_xys[:,2],inputs[:,2]],"g-")
    ######

def distance_loss(pred, targ):
    
    x = ((pred - targ)**2).sum(dim=-1)
    return x.mean()

#########################################

class NeuralNetworkStack(nn.Module):

    def __init__(self, n_in, n_out, nlayers=8, depth=20):
        super().__init__()
                
        Mod_list = []
        for i in range(nlayers):
            block = linear_layer(depth)
            Mod_list.append( block )
        
        self.blocks = nn.ModuleList(Mod_list)

        self.layer_first = nn.Linear(n_in, depth)
        self.layer_last = nn.Linear(depth*(nlayers+1)+n_in, n_out)
        
    def forward(self, x_in):
                
        x_next = self.layer_first(x_in)
    
        out_list = [x_in,x_next]
        for n, block in enumerate(self.blocks):
            x_next = block(x_next) 
            out_list.append(x_next)  
            
        x = torch.cat(out_list, dim=1)
        
        if self.training:
            x = F.leaky_relu(x)
            #x = 1-F.leaky_relu(1-x)
        else:
            x = torch.clamp(x,0)

        out = self.layer_last(x)
        
        return out
    
class NeuralNetworkSkip(nn.Module):
    def __init__(self, n_in, n_out, nlayers=8, depth=20):
        super().__init__()
                
        Mod_list = []
        for i in range(nlayers):
            block = linear_layer(depth)
            Mod_list.append( block )
        
        self.blocks = nn.ModuleList(Mod_list)

        self.layer_first = nn.Linear(n_in, depth)
        self.layer_last = nn.Linear(depth, n_out)
        
    def forward(self, x_in):
    
        x = self.layer_first(x_in)
        for n, block in enumerate(self.blocks):
            x = block(x)   
        out = self.layer_last(x)   
        
        return out
       
class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out,  n_layers):
        super(NeuralNetwork, self).__init__()

        d = 50
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_in, d),  nn.LeakyReLU(),
            nn.Linear(d, d),   nn.LeakyReLU(),
            nn.Linear(d, d),   nn.LeakyReLU(),
            nn.Linear(d, d),   nn.LeakyReLU(),
            nn.Linear(d, n_out),      
        )
    
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
    
def linear_layer(depth):
    return nn.Sequential( nn.Linear(depth, depth*2),   nn.LeakyReLU(),
                          nn.Linear(depth*2, depth),   nn.LeakyReLU()
    )
     