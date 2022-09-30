
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from functools import partial
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from functools import partial

from scipy.spatial.transform import Rotation as R

#############################################

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
     
##############################################
class kinematics():
    
    def __init__(self,a,t,r,d, active, limits):
        self.alpha = a
        self.theta = t
        self.radius = r 
        self.dists = d
        self.active = active
        self.limits = limits 
        self.orientation = False
        self.update_position()    

    def update_position(self):
        
        DH = self.get_motor_positions()
        self.position = forward_all(*DH)         

        return self.position
    
    def run_forward(self):

        DH = self.get_motor_positions()
        position = forward_all(*DH) 
        end_pos = position[:3,-1]    

        if self.orientation:
            DH = self.get_motor_positions()
            Tfs = make_transforms(*DH)[-1]
            orient = R.from_matrix(Tfs[:3,:3]).as_euler("XYZ")

            if orient[0]<0:  
                orient[[0,2]] = orient[[0,2]] + np.pi
                orient[[1]]   = np.pi - orient[[1]] 

            orient = np.degrees(orient)
            end_pos = np.append(end_pos,orient)

        return end_pos

    
    def random_positions(self,):
        
        mot_pos = self.get_motor_positions()
        active = self.active
        limits = self.limits
        
        for n,limit in enumerate(limits):
            if active[n]=="": continue
            if len(limit)==0: continue

            v = np.random.uniform(limit[0],limit[1])

            if active[n]=="t":            mot_pos[1,n] = v
            if active[n]=="r":            mot_pos[2,n] = v

        return mot_pos
    
    def get_motor_positions(self):
        
        alpha = self.alpha
        theta = self.theta
        radius = self.radius
        dists = self.dists
        active = self.active
        limits = self.limits

        mot_pos = np.array([alpha,theta,radius,dists])
        return mot_pos
        
    def generate_maps(self,n=100):
        
        inputs, target = [],[]
        for x in range(n):

            mot_pos = self.random_positions()
            self.set_motor_positions(mot_pos)

            act_pos = self.get_active()
            end_pos = self.run_forward()

            inputs.append(end_pos)
            target.append(act_pos)

        inputs = np.array(inputs)
        target = np.array(target)    

        return inputs, target

    def get_active(self):
        
        mot_pos = self.get_motor_positions()
        active = self.active

        out_pos = []
        for n,a in enumerate(active):
            if a=="": continue
            if a=="t":          v = mot_pos[1,n]    
            if a=="r":          v = mot_pos[2,n]
            #else: continue
            out_pos.append(v)

        return np.array(out_pos)

    def set_motor_positions(self, mot_pos):
        
        a,t,r,d = mot_pos
            
        self.alpha = a
        self.theta = t
        self.radius = r 
        self.dists = d
            
    def set_active(self, new_active):
        
        mot_pos = self.get_motor_positions()
        active = self.active

        i = 0
        for n in range(mot_pos.shape[1]):
            if active[n]=="": continue
            v = new_active[i]

            if active[n]=="r":            mot_pos[2,n] = v
            elif active[n]=="t":          mot_pos[1,n] = v
            else: continue
            i += 1
            
        self.set_motor_positions(mot_pos)
            

        return mot_pos



class with_torch():

    def forward_from_active(self, For_model, motor_pos, orientation=None):

        if orientation is None:   orientation = For_model.orientation

        pred_xys = []
        for p in motor_pos:
            DH = self.set_motors(For_model, p)

            pred_xy = self.forward_all(DH, orientation=orientation)
            pred_xys.append(pred_xy)

        pred_xys = torch.stack(pred_xys)

        return pred_xys

    def forward_all(self, DH, orientation=False):

        alpha, theta, radius, dists = DH
        
        alpha = torch.deg2rad(alpha)
        theta = torch.deg2rad(theta)

        TFs = self.make_transforms(alpha, theta, radius, dists)

        A = torch.Tensor((0,0,0,1))[None,:]
        DH = TFs[-1]

        positions = torch.matmul(DH,A.T).T
        positions = positions[:,:3]

        if orientation:
            rot = self.torch_mat_to_rot( DH[:3,:3] )
            if rot[[0]]<0:  
                    rot[[0,2]] = rot[[0,2]] + np.pi
                    rot[[1]]   = np.pi - rot[[1]] 
            rot = torch.rad2deg(rot)[None,]
            positions = torch.cat((positions,rot),dim=-1)

        return positions

    def make_transforms(self, alpha, theta, radius, dists):
        
        mot_params = torch.stack([alpha, theta, radius, dists])
        
        Tfs = []
        T = torch.eye(4)
        for i in range(mot_params.shape[1]):
            a,t,r,d = mot_params[:,i]
            DHt = self.DH_transform(a,t,r,d) 
            T = torch.matmul(T,DHt)     
            Tfs.append(T)

        return Tfs

    def DH_transform(self,a,t,r,z):
        Zt = self.Z_transform(a,t,r,z)
        Xt = self.X_transform(a,t,r,z)
        DHt = torch.matmul(Zt,Xt)
        return DHt

    ###################################
    def torch_mat_to_rot( self,  matrix ):

        central_angle = torch.asin( matrix[..., 0, 2] * (-1.0 if 0 - 2 in [-1, 2] else 1.0) )

        o = ( _angle_from_tan( "X", "Y", matrix[..., 2],   False, True ),
              central_angle,
              _angle_from_tan( "Z", "Y", matrix[..., 0,:], True,  True ))

        rotation = torch.stack(o, -1)
        return rotation

    ###############################################
    def X_transform(self,a,t,r,d):
        
        Xt = torch.eye(4)
        Xt[0,3] = r
        Xt[1,1] = torch.cos(a)
        Xt[1,2] = -1*torch.sin(a)
        Xt[2,1] = torch.sin(a)
        Xt[2,2] = torch.cos(a)
        
        return Xt

    def Z_transform(self,a,t,r,d):
        
        Zt = torch.eye(4)
        Zt[0,0] = torch.cos(t)
        Zt[0,1] = -1*torch.sin(t)
        Zt[1,0] = torch.sin(t)
        Zt[1,1] = torch.cos(t)
        Zt[2,3] = d
        
        return Zt

    def set_motors(self,For_model, new_motor):
        
        alpha  = torch.tensor(1.*np.array(For_model.alpha)) 
        theta  = torch.tensor(1.*np.array(For_model.theta)) 
        radius = torch.tensor(1.*np.array(For_model.radius))
        dists  = torch.tensor(1.*np.array(For_model.dists)) 
        active = For_model.active
        DH = torch.stack([alpha, theta, radius, dists])

        i = 0
        for n in range(DH.shape[1]):
            if active[n]=="": continue
            v = new_motor[i]

            if   active[n]=="r":          DH[2,n] = v
            elif active[n]=="t":          DH[1,n] = v
            else: continue
            i += 1

        return DH

def _angle_from_tan( axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool ):

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    
    if horizontal: i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:   return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:           return torch.atan2(-data[..., i2], data[..., i1])
    
    return torch.atan2(data[..., i2], -data[..., i1])

############################################
def train_motors(For_model, model):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for n in range(100):
        print("-------------")
        inputs, target = For_model.generate_maps(5000)
        for _ in range(5):
            train_loop(model, inputs, target, distance_loss, optimizer)
        #if n%10==0: optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.8    

def train_loop(model, inputs, target, loss_fn, n_reps = 10, optimizer=None):
    
    if optimizer is None:  optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sumloss = 0 
    model.train()
    n_reps = 5
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
    
def train_positions(For_model, model):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    For_model
    n_reps = 5000
    for n in range(n_reps):
        inputs, target = For_model.generate_maps(500)
        inputs = torch.Tensor(inputs)

        pred = model(inputs)

        if inputs.shape[-1]==3:         orientation=False
        else:                           orientation=True
     
        pred_xys = with_torch().forward_from_active(For_model, pred, orientation=orientation)
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
     
    pred_xys = with_torch().forward_from_active(For_model, pred, orientation=orientation)

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
    return x.mean()**0.5
        
######################################

def get_DH_params(model):
    
    alpha = model["alpha"]
    theta =  model["theta"]
    radius = model["radius"]
    dists = model["dists"]
    active = model["active"]
    limits = model["limits"]
    
    return alpha, theta, radius, dists, active, limits
    
##############################################################

def update_position(mot_pos, new_active, active):

    mot_pos = set_active(mot_pos, new_active, active)
    Positions = forward_all(*mot_pos)  

    return Positions

def generate_maps(alpha, theta, radius, dists, active, limits):
    
    inputs, target = [],[]
    for x in range(500):
        mot_pos = random_positions( alpha, theta, radius, dists, active, limits)
        act_pos = get_active(mot_pos, active, limits)

        Positions = forward_all(*mot_pos)[:,:3]
        EndPos = Positions[-1]

        inputs.append(EndPos)
        target.append(act_pos)

    inputs = np.array(inputs)
    target = np.array(target)    
    
    return inputs, target

def random_positions( alpha, theta, radius, dists, active, limits):
    
    mot_pos = np.array([alpha,theta,radius,dists])
    
    for n,limit in enumerate(limits):
        if active[n]=="": continue
        if len(limit)==0: continue

        v = np.random.uniform(limit[0],limit[1])

        if active[n]=="t":            mot_pos[1,n] = v
        if active[n]=="r":            mot_pos[2,n] = v
    
    return mot_pos
             
def get_active(mot_pos, active):
    
    out_pos = []
    for n,a in enumerate(active):
        if a=="": continue
        if a=="t":          v = mot_pos[1,n]    
        if a=="r":          v = mot_pos[2,n]
        #else: continue
            
        out_pos.append(v)
    
    return np.array(out_pos)

def set_active(mot_pos, new_active, active):
    
    i = 0
    for n in range(mot_pos.shape[1]):
        if active[n]=="": continue
        v = new_active[i]

        if active[n]=="r":            mot_pos[2,n] = v
        elif active[n]=="t":          mot_pos[1,n] = v
        else: continue
        i += 1
            
    return mot_pos
#######################
################################################

def forward_all(alpha, theta, radius, dists):

    DHts = make_transforms(alpha, theta, radius, dists)

    A = np.array((0,0,0,1))[None,:]
    positions = []
    for DH in DHts:
        B = np.matmul(DH,A.T).T   
        positions.append(B)

    positions = np.array(positions)
            
    return positions[:,0,:3].T

def make_transforms(alpha, theta, radius, dists):
    
    DHts = []
    T = np.eye(4)
    for i in range(len(alpha)):
        a = np.radians(alpha[i])
        t = np.radians(theta[i])
        r = radius[i]
        d = dists[i]
        DHt = DH_transform(a,t,r,d) 
        DHt = np.matmul(T,DHt)
        T = DHt.copy()
        DHts.append(DHt)

    return np.array(DHts)

def DH_transform(a,t,r,z):
    Zt = Z_transform(a,t,r,z)
    Xt = X_transform(a,t,r,z)
    DHt = np.matmul(Zt,Xt)
    return DHt

###############################################
def X_transform(a,t,r,d):
    
    Xt = [[1,0,0,r],
          [0,np.cos(a), -1*np.sin(a),0],
          [0,np.sin(a),  np.cos(a),0],
          [0,0,0,1]]
    Xt = np.array(Xt)

    return Xt
    
def Z_transform(a,t,r,d):
    
    Zt = [[np.cos(t), -1*np.sin(t),0,0],
          [np.sin(t), np.cos(t), 0,0],
          [0,0,1,d],
          [0,0,0,1]]
    Zt = np.array(Zt)
    return Zt

######################################
def animate_robot(alpha,theta, radius, dists, active, limits):

    fig = plt.figure()
    for n,limit in enumerate(limits):
        if active[n]=="": continue
        if len(limit)==0: continue

        values = np.linspace(limit[0],limit[1],10)
        for v in values:
            if active[n]=="r":            radius[n] = v
            if active[n]=="t":            theta[n] = v

            Positions = forward_all(alpha, theta, radius, dists)
            plot_joints(Positions,fig=fig)
            plt.pause(0.01)

def plot_joints(positions, fig=None):
    
    if fig is None: fig = plt.figure()
        
    x,y,z = positions
    n = range(len(x))
    ax = fig.axes
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    ax.plot3D(x, y, z, 'gray', linewidth=5)
    ax.scatter3D(x, y, z, c=n, s= 100, cmap='hsv')
    plot_plane(ax)
    plot_axes(ax)
    
def plot_plane(ax):
    
    x = np.linspace(-100, 100, 10)
    y = np.linspace(-100, 100, 30)
    X, Y = np.meshgrid(x, y)
    Z = X*Y*0
    ax.plot_surface(X, Y, Z, alpha=0.2)
    #ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
def plot_axes(ax):
    zero = [0,0]
    x = [0,100]
    y = [0,100]
    z = [0,100]
    ax.plot3D(x, zero, zero, 'red')
    ax.plot3D(zero, y, zero, 'green')
    ax.plot3D(zero, zero, z, 'blue')

#######################################
preset_models = {
    "Prismatic3": { ## Full Prismatic 3
        "alpha"  : [0,90,90],        "theta"  : [0,90,90] ,
        "radius" : [200,140,150],    "dists"  : [0,0,0],
        "active" : ["r","r","r"],    "limits" : [[0,200],[0,200],[0,200]]
        },  ## To do: Make the end effector point down, translate in z,x,y     
    
    "Revolute":{## Full Revolute 3
        "alpha"  : [90,180,0],         "theta" : [ 45,40,30] ,
        "radius" : [20,75,50],       "dists" : [0,0,0],
        "active" : ["t","t","t"],    "limits" : [[-10,370],[0,180],[0,180]]    
        },
    
    "Columninar4":{
        "alpha": [ 90,180,90,0,0],       "theta" : [ 0,90,90,25,-45],
        "radius" : [10,50,0,25,25],      "dists" : [0,0,0,0,0],
        "active" : ["t","r", "","t","t"],  "limits" : [[0,360],[0,100],[],[0,180],[0,180]]
        },
    "Standford":{
        "alpha":   [0,45,0,45,45,45], "theta" : [-90,90,0,-90,90,0],
        "radius" : [0,35,20,0,0,15],  "dists" : [0,0,0,0,0,0],
        "active" : ["t","r", "","t"],  "limits" : [[0,360],[0,100],[],[0,180]]
        },
    "KUKA":{
        "alpha":   [90,0,180,0,0,90],    "theta" : [35,70,-85,25,45,25],
        "radius" : [20,70,50,5,5,25],    "dists" : [0,0,0,0,0,0],
        "active" : ["t","r", "","t"],  "limits" : [[0,360],[0,100],[],[0,180]]
        },

    "HangingArm":{
        "alpha"  :[0,180, 90,0,-90],  "theta" : [0,90, 45,40,0],
        "radius" :[0,100,10,75,50],    "dists" : [200,0,0,0,0],
        "active" :["","r","t","t","t"],"limits" : [[],[20,200],[0,270],[0,180],[0,180]]
        }
    }

##

