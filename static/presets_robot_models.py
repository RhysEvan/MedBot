#######################################
# for active t stands for theta so means a rotational joint
# for active r stands for radius so means a translation movement

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
        "alpha": [ 90,180,90,0,0],       "theta" : [ 0,90,90,25,-45] ,
        "radius" : [10,50,0,25,25],      "dists" : [0,0,0,0,0],
        "active" : ["t","r", "","t","t"],  "limits" : [[0,360],[0,100],[],[0,180],[0,180]]
        },
    "Stanford":{
        "alpha":   [0,45,0,45,45,45], "theta" : [-90,90,0,-90,90,0] ,
        "radius" : [0,35,20,0,0,15],  "dists" : [0,0,0,0,0,0],
        "active" : ["r","r","r","r","r","r"],  "limits" : [[30,360],[30,360],[30,360],[30,360],[30,360],[0,180]]
        },
    "KUKA":{
        "alpha":   [90,0,180,0,0,90],    "theta" : [35,70,-85,25,45,25],
        "radius" : [10,30,20,5,5,25],    "dists" : [0,0,0,0,0,0],
        "active" : ["t","t","t","t","t","t"],  "limits" : [[30,360],[30,360],[30,360],[30,360],[30,360],[0,180]]
        },

    "HangingArm":{
        "alpha"  :[0,0,180,90,0,-90],  "theta" : [0,0,90, 270,80,40] ,
        "radius" :[0,0,100,10,75,50],    "dists" : [0,200,0,0,0,0],
        "active" :["r","","r","t","t","t"], "limits" : [[0,200],[],[20,200],[0,270],[0,180],[0,180]]
        }, 

    "HangingArm1":{
        "alpha"  :[0,180,90,0,-90],  "theta" : [0,90, 45,40,0],
        "radius" :[0,100,10,75,50],    "dists" : [200,0,0,0,0],
        "active" :["","r","t","t","t"],"limits" : [[],[20,200],[0,270],[0,180],[0,180]]
        }

    




    }

