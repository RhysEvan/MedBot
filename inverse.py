import numpy as np
import torch


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