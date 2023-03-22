from retna.networks import *
from retna.training import *
from retna.data_loader import *
from retna.view import *
from retna.prune import *
from retna.hook_tools import *
from retna.simulate_data import *

import torch
import glob as glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.transform import resize
from skimage import img_as_ubyte

class Main():
    def __init__(self, 
                 path_data, 
                 path_model, 
                 chans, 
                 size=[100,100], 
                 label=False, 
                 noise=False, 
                 batch=1,
                 rnd_noise=0.3, 
                 noise_range=[.1,.5],
                 ns_par=[0,0.3,1]):
        
        self.path_data = path_data
        self.path_model = path_model
        self.h_chans = chans
        self.force_noise = noise
        self.batch = batch
        self.size = size
        self.label = label
        self.rand_noise = rnd_noise
        self.range_noise = noise_range
        self.ns_param = ns_par
    
    def training(self, model, num, load=False, loss = 10, train = 20, cycles = 20, patience = None, photo=False):
        if load == True:
            print("getting model")
            self.call_trained_full_model(self.path_model)
        else:
            self.model = Retna_V1(3,5, self.h_chans)
            self.optimizer = None
            self.epoch = 0
        self.loader()

        self.model = self.model.to("cuda:0")
        self.Loader.dataset.device = "cuda:0"
        self.model, self.optimizer, self.epoch = train_model(self.model, 
                                                             model,
                                                             self.Loader, 
                                                             path_model=self.path_model,
                                                             optimizer=self.optimizer, 
                                                             cur_epoch=self.epoch, 
                                                             num_epochs=num, 
                                                             n_select_loss=loss, 
                                                             n_select_train=train, 
                                                             n_train_cycles=cycles, 
                                                             threshold = patience,
                                                             check_photo=photo)
        #self.model.save("train")
        checkpoint = {
            'model': self.model, 
            'optimizer': self.optimizer,
            'epoch': self.epoch
            }
        torch.save(checkpoint,self.path_model+model)

    def call_trained_full_model(self, path, model):
        info = torch.load(path+model)
        self.model = info['model']
        self.optimizer = info['optimizer']
        self.epoch = info['epoch']

    def save_state_dict(self, model):
        torch.save(self.model.state_dict(), self.path_model+model)
        #torch.save({
        #        'epoch': self.epoch,
        #        'model_state_dict': self.model.state_dict(),
                #'optimizer_state_dict': self.optimizer.state_dict(),
                #'loss': self.loss
        #        }, self.path_model+model)
    
    def load_state_dict(self,path, model):
        self.model = Retna_V1(3,5, [32,32,16,16,16,8])
        self.model.load_state_dict(torch.load(path+model))
        #checkpoint = torch.load(self.path_model+model)
        #self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.epoch = checkpoint['epoch']
        #self.optimizer = None
        self.model.eval()

    def loader(self):
        self.datas =  glob.glob(self.path_data)
        self.datas = Image_Handler(self.datas)
        self.Loader = DataLoader(self.datas, batch_size=self.batch)
        #####################parameters#####################
        self.Loader.dataset.outsize = self.size
        self.Loader.dataset.force_label = self.label
        self.Loader.dataset.noise = self.force_noise
        self.Loader.dataset.nois_low = self.ns_param[0]
        self.Loader.dataset.noise_high = self.ns_param[1]
        self.Loader.dataset.noise_size = self.ns_param[2]
        self.Loader.dataset.scale_range = [.2,.8]
        self.Loader.dataset.noise_range = self.range_noise

    def mosaic(self):
        self.loader()
        print_mosaic(self.Loader,self.model)
        plt.show()

    def cam_predict(self, crop, path, model):
        self.load_state_dict(path, model)
        if self.model is not None:  self.model = self.model.to("cuda:0")
        crop = torch.from_numpy(resize(crop, (150,150)))
        crop = crop.permute(2,0,1)
        crop = torch.unsqueeze(crop,0)
        crop = crop.type(torch.cuda.FloatTensor)
        pred = self.model(crop)
        P = colorize_channels(pred)
        return P

if __name__ == "__main__":
    data_path = r"C:\Users\mheva\OneDrive\Bureaublad\temp/*"
    model_path = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\Retna\models"
    model = "\\pleora_state_dict_3.pt"
    call = Main(data_path,
                model_path,
                [32,32,16,16,16,8],
                size=[150,150],
                label=True,
                batch=1,
                rnd_noise=0.3,
                noise=True,
                ns_par=[0,0.3,1])
    
    #for i in range(1):
    #    try: call.training(model, 50, True, loss=50, train=200, cycles=10, patience = None, photo=False)
    #call.training(model, 10000, False, loss=80, train=10, cycles=10, patience = 80, photo=False)
    call.load_state_dict(model)
    #call.call_trained_full_model(model_path, model)
    call.mosaic()


#desirable idea, creating a GUI that you are easily able to switch between:
# model type
# does the database need to have a label in it yes or no?
# how many cycles?
# how many train selects?
# how many loss selects?
# how many epochs
# what batch size? doesn't work well right now.
# how many hidden channels
# what is the base size of the data loaded (you can decide it yourself within the space of the picture base recommend is 150x150)
# explorer window to find the folder with the database
# explorer window to find the pt file IF IT EXISTS IN THE FIRST PLACE!!! MAKE A CHECK FOR REDUNDANCY!!
# a select window with all sorts of options of manipulations like:
# skew - more difficult then it was expected to apply
# scale
# noise
# many others
# give a information window or make it so that a button opens a txt file with all the necessary information about the functions and or
# general information about the structure of the AI code.