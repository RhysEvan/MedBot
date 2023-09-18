
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import torch
import Inverse_Kinematics.inverse as inverse

def train_motors(For_model, model):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for n in range(100):
        inputs, target = For_model.generate_maps(1000)
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

        if n%10==0: optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.5  
        #if n%100==0: optimizer.param_groups[0]["lr"] = 1e-3  

def train_positions_paths(For_model, model, n_cycles = 500):

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    first = True
    second = False
    For_model
    divide = 1
    memory = 0
    lock = False
    val_loss = 10000
    for n in range(n_cycles):
        n += 1
        if not second:
            inputs, target = For_model.generate_paths(3000)
        else:
             inputs, target = For_model.generate_paths(4000)
        inputs = torch.Tensor(inputs)
        pred = model(inputs)

        if inputs.shape[-1]==3:         orientation=False
        else:                           orientation=True
     
        pred_xys = inverse.with_torch().forward_from_active(For_model, pred, orientation=orientation)
        pred_xys = pred_xys[:,-1,:]
        if first or val_loss > 2 and not second:
            loss = distance_loss(pred_xys, inputs).mean()
        else:
            return
            loss = total_loss(pred_xys, inputs).mean()
            second = True
            if not lock:
                low_loss = val_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        val_loss = np.array(loss.item()).round(2)   

        if first or second and not lock:
            print(np.array(loss.item()).round(2)) 
            low_loss = val_loss
            first = False
            if second:
                lock = True

        elif low_loss>val_loss:
            print("{}  caching model improvement".format(np.array(loss.item()).round(2)))
            cach = model
            low_loss = val_loss
            memory = 0
        else:
            print(np.array(loss.item()).round(2))
            memory +=1
            if memory%25 ==0:
                optimizer.param_groups[0]["lr"] = 1e-3
                print("{} making lr bigger {}".format(np.array(loss.item()).round(2), optimizer.param_groups[0]["lr"]))
            elif memory%5==0:
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.5 
                print("{} making lr smaller {}".format(np.array(loss.item()).round(2), optimizer.param_groups[0]["lr"]))
                
            if optimizer.param_groups[0]["lr"]<1e-15:
                optimizer.param_groups[0]["lr"] = 1e-3
        #if n%10==0: optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.5 
        #if n%100==0: optimizer.param_groups[0]["lr"] = 1e-3  

def evaluate_plot(For_model, model):
    
    inputs, target = For_model.generate_maps(100)
    inputs = torch.Tensor(inputs)
    model.eval()

    pred = model(inputs)

    if inputs.shape[-1]==3:         orientation=False
    else:                           orientation=True
     
    pred_xys = inverse.with_torch().forward_from_active(For_model, pred, orientation=orientation)
    pred_xys = pred_xys[:,-1,:]
    print(distance_loss(pred_xys, inputs))

    pred_xys = np.array(pred_xys.detach()).squeeze()
    inputs = np.array(inputs)

    fig,axs = plt.subplots(1,2)
    axs[0].plot(pred_xys[:,0],pred_xys[:,1],"ro")
    axs[0].plot(inputs[:,0],inputs[:,1],"bo")
    axs[0].plot([pred_xys[:,0],inputs[:,0]], [pred_xys[:,1],inputs[:,1]],"g-")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("red = prediction")

    axs[1].plot(pred_xys[:,0],pred_xys[:,2],"ro")
    axs[1].plot(inputs[:,0],inputs[:,2],"bo")
    axs[1].plot([pred_xys[:,0],inputs[:,0]], [pred_xys[:,2],inputs[:,2]],"g-")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    axs[1].set_title("blue = target")
    if inputs.shape[-1]!=3:
        fig,axs = plt.subplots(1,3)
        axs[0].plot(pred_xys[:,3],inputs[:,3], "o")
        axs[0].set_xlabel("a")
        axs[0].set_ylabel("a")

        axs[1].plot(pred_xys[:,4],inputs[:,4], "o")
        axs[1].set_xlabel("b")
        axs[1].set_ylabel("b")
        axs[1].set_title("euler angles")
        
        axs[2].plot(pred_xys[:,5],inputs[:,5], "o")
        axs[2].set_xlabel("g")
        axs[2].set_ylabel("g")
        axs[2].set_title("blue = target")


    ######

def total_loss(pred, targ, theta):
    x1 = distance_loss(pred, targ)
    x2 = theta_loss(theta)
    return .01*x2+x1

def distance_loss(pred, targ): 
    #print("pred")
    #print(pred.round())
    #print("targ")
    #print(targ.round())
    cart_pred = pred[:,:3]
    cart_targ = targ[:,:3]
    eul_pred = pred[:,3:]
    eul_targ = targ[:,3:]
    dif_cart = ((cart_pred - cart_targ)**2).sum(dim=-1)
    dif_eul = compare_angles(eul_pred, eul_targ)
    #print("dif cart")
    #print(dif_cart)
    #print("dif eul")
    #print(dif_eul)
    x = (dif_cart*1 + dif_eul*0)
    return x.mean()

def theta_loss(pred):
    #input is the list with all theta values predicted by the ai.
    pred_next = pred[1:]
    pred_prev = pred[:-1]
    x = ((pred_next - pred_prev)**2).sum()
    return x.mean()

def compare_angles(angle_pred, angle_targ):
    diff = torch.abs(angle_pred - angle_targ)
    diff[diff>180] = 360 - diff[diff>180]
    result = (diff**2).sum(dim=-1) 
    return result

#########################################
class NeuralNetworkBlind(nn.Module):
    def __init__(self, input, output, nlayers = 8, width=20):
        super().__init__()
        Mod_list = []
        for i in range(nlayers):
            block = linear_deep(width*2**(i+1))
            Mod_list.append( block )
        self.blocks = nn.ModuleList(Mod_list)
        self.layer_first = nn.Linear(input, width*2)
        self.layer_last = nn.Linear(width*nlayers*2**(nlayers-1)+input-2*width, output)
        
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
        

class NeuralNetworkStack(nn.Module):

    def __init__(self, n_in, n_out, nlayers=8, width=20):
        super().__init__()
                
        Mod_list = []
        for i in range(nlayers):
            block = linear_layer(width)
            Mod_list.append( block )
        
        self.blocks = nn.ModuleList(Mod_list)
        self.layer_first = nn.Linear(n_in, width)
        self.layer_last = nn.Linear(width*(nlayers+1)+n_in, n_out)
        
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
    def __init__(self, n_in, n_out, nlayers=8, width=20):
        super().__init__()
                
        Mod_list = []
        for i in range(nlayers):
            block = linear_layer(width)
            Mod_list.append( block )
        
        self.blocks = nn.ModuleList(Mod_list)

        self.layer_first = nn.Linear(n_in, width)
        self.layer_last = nn.Linear(width, n_out)
        
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
    
def linear_layer(width):
    return nn.Sequential( nn.Linear(width, width*2),   nn.LeakyReLU(),
                          nn.Linear(width*2, width),   nn.LeakyReLU()
    )

def linear_deep(width):
    return nn.Sequential( nn.Linear(width, width*2), nn.LeakyReLU(),
                          nn.Linear(width*2,width*4), nn.LeakyReLU(),
                          nn.Linear(width*4,width*2), nn.LeakyReLU(),)

def get_DH_params(model):
    
    alpha = model["alpha"]
    theta =  model["theta"]
    radius = model["radius"]
    dists = model["dists"]
    active = model["active"]
    limits = model["limits"]
    
    return alpha, theta, radius, dists, active, limits