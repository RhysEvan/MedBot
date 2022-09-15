"""turn off sciview in File -> Settings.... -> Tools -> SciView"""
# Import all needed libraries

from cProfile import label
import os

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon2mask, disk
from skimage import io
from copy import deepcopy
import cv2
import easygui as eg
import pandas as pd
import glob

# This menu is called when running the file, it opens the context windows to choose your path
class menu():
    def __init__(self):
        self.folder = ""
        self.excell = ""
        self.labels = []

        self.mainmenu()
    #open mainmenu, depending on what option the user picks, that new menu is opened
    def mainmenu(self):
        event = eg.indexbox("Please select the working directory and excel folder or use previous configuration",
                            "Annotator",["Choose file path","Choose excell path","Previous Config", "Done","About"])
        if event == 0:
            self.pathmenu()
        elif event == 1:
            self.excellmenu()
        elif event == 3:
            pass
        elif event == 4:
            self.about()
        elif event == 2:
            self.labels = [] #to prevent two tabels if the user first selects an excel file, then chooses use prev
            with open(str(os.getcwd())+'\\prefs.txt') as f:
                lines = f.readlines()
            self.excell = lines[4].strip('\n')
            self.folder = lines[5].strip("'\n")
            df = pd.read_excel(self.excell,header=None)
            dflist = df[0].tolist()
            for i in range(len(dflist)):
                self.labels.append([dflist[i]])
            self.labels.append([str(len(self.labels)+1)+'-40: Unused labels'])
            pass
        
    def excellmenu(self):
        self.excell = eg.fileopenbox("Choose excelldocumentje") #opens your computers directory and saves the chosen path
        df = pd.read_excel(self.excell,header=None)
        dflist = df[0].tolist()
        for i in range(len(dflist)): #Read out the excel and puts the lines in lists in 1 big list (format plt.table)
            self.labels.append([dflist[i]])
        self.mainmenu() #go back to the main menu
 

    def pathmenu(self):
        self.folder = eg.diropenbox("Choose directory") #opens your computers dierectory and saves the chosen path
        self.mainmenu() #go back to the main menu
    
    def about(self):
        with open('Read_Me.txt') as f:
            lines = f.read()
        eg.msgbox(lines,ok_button="Next")
        self.mainmenu() #go back to the main menu

###############################
#  Annotator 
###############################

class Annotate():

    def __init__(self, exclude_labeled=False, run=True, bands=[1]):
        
        self.m = menu()
        self.excell = self.m.excell
        self.searchpath = self.m.folder + r'/*'
        filenames = glob.glob(self.searchpath)
        print(filenames)
        #Excluded labeled = True will ignore H5 files in the map that contain "_labeled"
        if exclude_labeled and '.h5' in filenames[0]:
            filenames = [fn for fn in filenames if not os.path.exists(fn.replace("_hypercube", "_labeled_hypercube"))]
            filenames = [fn for fn in filenames if fn.find("_labeled_hypercube") == -1]
        if len(filenames) == 0:
            print("No images")
            return

        self.layers = bands
        self.band = 0

        self.filenames = filenames
        self.im_idx = 0
        self.cur_idx = 0
        self.fill = 0.25
        self.drawn = []

        self.model = None
        self.draw_label = False

        self.polygons = [[] for i in range(len(filenames))]
        self.cur_polygons = []

        self.nbands = None

        with open(str(os.getcwd())+'\\prefs.txt') as f:
                lines = f.readlines()
        self.labelreset = lines[3].strip("\n") #readlines returns \n too, strip removes this
        #layout
        self.instructionsloc = 'right'
        self.labelsloc = 'left'
        self.filenameloc = 'top'
        try:
            with open(str(os.getcwd())+'\\prefs.txt') as f:
                lines = f.readlines()
            self.instructionsloc = lines[0].strip("'\n")
            self.labelsloc = lines[1].strip("'\n")
            self.filenameloc = lines[2].strip("'\n")
        except:
            pass
        #This run starts the program
        if run:
            self.open_window()
    #Opens a new matplotlib subplot and connects the script to the canvas, if an event happens in the canvas (such as clicking) this gets detected
    def open_window(self):

        plt.rcParams['keymap.fullscreen'] = []
        fig, ax = plt.subplots(figsize=(15, 12))
        self.fig = fig
        self.ax = ax
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.fig.canvas.mpl_connect('key_press_event', self.onKey)
        self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.draw_image()

    def on_close(self,event):
        self.polygons[self.im_idx] = self.cur_polygons

    def read_image(self, filename):
        im = None
        if ".jpg" in filename.lower() or ".tif" in filename.lower() or ".png" in filename.lower():
            im = cv2.imread(filename)
        if "hypercube.h5" in filename.lower():
            im = read_H5_hypercube(filename, all=self.layers)
        elif ".h5" in filename.lower():
            im = read_H5(filename)
        return im

    def load_labeled(self):
        filename = self.filenames[self.im_idx]
        im = None
        if ".jpg" in filename.lower() or ".tif" in filename.lower():
            im = cv2.imread(filename)
        if "hypercube.h5" in filename.lower():
            im = read_H5_hypercube(filename, dataset="labels")
        elif ".h5" in filename.lower():
            im = read_H5(filename, dataset="labels")
        if im is not None:
            if im.dtype == np.bool:        im = im * 255
            if im.ndim == 2:             im = im[..., None]
            im = im.astype(np.uint8)
        if im.ndim == 3:
            self.bands = im.shape[2]
        else:
            self.bands = 1
        return im

    def load_data(self):
        filename = self.filenames[self.im_idx]
        im = self.read_image(filename)
        if im is None:
            print(str(self.im_idx) + ": Not Found")
            self.filenames.pop(self.im_idx)
            self.polygons.pop(self.im_idx)
            im = self.load_data()
        if im.dtype == np.bool_:        im = im * 255
        if im.ndim == 2:             im = im[..., None]
        im = im.astype(np.uint8)
        if im.ndim == 3:
            self.bands = im.shape[2]
        else:
            self.bands = 1
        return im

    def draw_image(self):
        hc = self.load_data()
        im = hc[:, :, self.band]  # choose band in case of hyperspectral data, band is a list, THIS IS THE ONE YOU DRAW LATER ON
        self.ax.clear()
        if im.shape[-1] == 1:
            im = im * [1, 1, 1]
        im = im.astype(np.uint8)
        self.ax.imshow(im)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        #tables (from here back in use) This part draws tables next to the image
        #instructions table
        instr = [["→: Next Image"], ["←: Last Image"],
                ["↓: Next label"], ["↑: Previous label"],
                ["1: Next band"], ["2: Previous band"],
                ["Backspace: Remove Point"], ["Enter/Right Click: Next Object"],
                ["f: Toggle fill opacity " + "(" + str(self.fill) + ")"], ["i: More information"],["o: Options"],["Escape: Close Tool"]]
        table_instr = plt.table(cellText=instr, loc=self.instructionsloc, cellLoc='left')
        table_instr.set_fontsize(12)
        table_instr.scale(0.5, 2.5)
        #current label and filename table
        fn_str = self.filenames[self.im_idx]
        fn_str = ("..." + fn_str[-50:] if len(fn_str) > 50 else fn_str)
        currentlabeltxt = [[fn_str],["Band: " + str(self.band)]]
        table_curlabel = plt.table(cellText=currentlabeltxt, loc=self.filenameloc, cellLoc='center')
        table_curlabel.set_fontsize(16)
        table_curlabel.scale(1, 3)
        #label legend table
        colors = [["w"]]*len(self.m.labels) #makes a list of lists (format for plt.table) with the color white
        colors[min(self.cur_idx,len(self.m.labels)-1)] =["#1ac3f5"] #changes the 'cur_idx'-th color white with blue to indicate the cur label
        table_legend = plt.table(cellText=self.m.labels, loc=self.labelsloc, cellLoc='right',cellColours=colors)
        table_legend.set_fontsize(10)
        table_legend.scale(0.5, 2)
    
    def writetoprefs(self):
        with open(str(os.getcwd())+'\\prefs.txt','w') as f:
            f.write(str(self.instructionsloc)+str("\n")+
                    str(self.labelsloc)+str("\n")+
                    str(self.filenameloc)+str("\n")+
                    str(self.labelreset)+str("\n")+
                    str(self.excell)+str("\n")+
                    str(self.m.folder))
            f.close
    #Called when pressing 'o' (see later how keypresses are handled) and opens a contextmenu with options, same structure as the initial menu that opens
    def optionsmain(self):
        event = eg.buttonbox("Options","Annotation Tool",["Label preferences","Layout","Done"])
        if event == "Label preferences":
            self.labelprefmenu()
        elif event == "Layout":
            self.layoutmenu()
        elif event == "Done":
            pass

    def labelprefmenu(self):
        self.labelreset = eg.boolbox("Reset label index when changing images?","Label preferences",["Reset index","Keep current index"],None,cancel_choice=False)
        self.optionsmain()
    
    def layoutmenu(self):
        locations = ['bottom','bottom left','bottom right',
                     'center','center left','center right',
                     'left','lower center','lower left',
                     'lower right','right','top','top left',
                     'top right','upper center','upper left',
                     'upper right']
        event = eg.buttonbox("Choose positions or reset to default.","Layout",["Default","Choose locations"])
        if event == "Choose locations":
            event = eg.choicebox("Choose a location for the label table (default 'left')\nThe location will update next time you change the image","Layout",locations)
            self.labelsloc = event
            locations.remove(event)
            event = eg.choicebox("Choose a location for the hotkey table (default 'right')\nThe location will update next time you change the image","Layout",locations)
            self.instructionsloc = event
            locations.remove(event)
            event = eg.choicebox("Choose a location for the current filename table (default 'top')\nThe location will update next time you change the image","Layout",locations)
            self.filenameloc = event
            self.optionsmain()
        elif event == "Default":
            self.labelsloc = 'left'
            self.instructionsloc = 'right'
            self.filenameloc = 'top'
            self.optionsmain()

    ###########################################
    ##  UI
    ###########################################
    #this secion handles keypresses, onKey is constantly being looked at in  open_window because we are connected to the canvas as stated in a prev comment
    def onKey(self, event):
        if event.key == 'right':
            if self.labelreset: #user option, makes it so that the current label (=cur_idx) remains or resets after changing images
                self.cur_idx = 0
            self.polygons[self.im_idx] = self.cur_polygons #the polygons that the user drew are added to the parent list 'polygons'
            self.change_image(1)
            self.draw_image #this is called everytime ANYTHING happens (switching image, adding a point,...) to refresh the image
            self.update()

        if event.key == 'left':
            if self.labelreset: #user option, makes it so that the current label (=cur_idx) remains or resets after changing images
                self.cur_idx = 0
            self.change_image(-1)
            self.draw_polygons() #Also a way to refresh the image
            self.draw_image #this is called everytime ANYTHING happens, won't be repeating this comment
            self.update()

        if event.key == 'down':
            self.cur_idx += 1 #current label + 1
            self.draw_image()
            self.submit_polygon() # copies the current polygon points to the parent list and resets the current polygon list
            self.draw_polygons()
            
        if event.key == 'up':
            self.cur_idx = max(0, self.cur_idx - 1) #current label - 1, cant be negative so check for the max between zero and idx-1
            self.submit_polygon()
            self.draw_image()
            self.draw_polygons()
            self.update()

        if event.key == 'enter': #Unused
            self.submit_polygon()

        if event.key == 'escape':
            self.polygons[self.im_idx] = self.cur_polygons
            plt.close(self.fig)

        if event.key == 'backspace':
            if len(self.cur_polygons[-1]["pts"]) > 0: #look at the last (-1) dictionairy in the string cur_polygons and looks at the key of the dict "pts", this holds the 
                #points of the CURRENTLY drawn polygon, if its len is greater than zero, so there are points drawn, then we can remove the last point
                self.cur_polygons[-1]["pts"].pop()
                #However if this dict is empty, that means that there are currently no points for the current polygon, we check the length of cur_polygon:
            elif len(self.cur_polygons) > 1: #if more than one dictionairy in the list
                self.cur_polygons.pop()
            self.draw_polygons()

        if event.key == "f":
            self.fill = (self.fill + 0.25) % 1 #mod, gives the remaining after division, this means that we can endlessly cycle over fill= 0; 0.25; 0.5; 0.75; 1; 0; 0.25;...
            self.draw_image()
            self.draw_polygons()
            self.update()
        
        if event.key == "i": #brings up the information window, Reads of the attached file "About.txt"
            with open('Read_Me.txt') as f:
                lines = f.read()
            eg.msgbox(lines,ok_button="Next") 
        
        if event.key == "o": #brings up the options menu as explained in a previous comment
            self.optionsmain()

        if event.key == "l": #unused
            self.draw_label = not self.draw_label
            self.draw_image()

        if event.key == "1": #switch to the previous image layer
            #For index errors:
            #If the current band as long as the current band is smaller than ("the bands the user"-1) wanted AND
            #smaller than ("the amount of available bands"-1), 1 is added to the band. (-1 because of index 0)
            if self.band < len(self.layers)-1 and self.band < self.bands-1:
                self.band = self.band+1
            else:
                self.band = min(len(self.layers)-1,self.bands-1)
            self.change_image()
            self.update()

        if event.key == "2": #switch to the next image layer
            self.band = max(self.band - 1, 0)
            self.change_image()
            self.update()

    def onClick(self, event):

        if event.button == 1:  ## Left click
            if event.inaxes:
                if len(self.cur_polygons) == 0: #if no polygons were drawn, we reset the index and points to their init value
                    #cur_polygons holds a dict with {"idx": the index, "pts": the drawn points}
                    self.cur_polygons = [{"idx": self.cur_idx, "pts": []}]
                label = self.cur_polygons[-1]
                L_idx = label["idx"]
                polygons = label["pts"]
                polygons.append([event.xdata, event.ydata]) #add the location where we clicked as a point to the polygons list
                self.cur_polygons[-1] = {"idx": L_idx, "pts": polygons}
                self.draw_polygons()
        if event.button == 3: #right-click
            self.submit_polygon()
    def onScroll(self, event):
        self.change_image(int(event.step))

    ###########################################
    ## Image stuff
    ###########################################
    def change_image(self, step=0):
        self.polygons[self.im_idx] = self.cur_polygons
        self.im_idx = (self.im_idx + step) % len(self.filenames)
        self.cur_polygons = self.polygons[self.im_idx]
        self.draw_image()

    def submit_polygon(self):
        new_poly = {"idx": self.cur_idx, "pts": []}
        if len(self.cur_polygons) == 0:
            self.cur_polygons.append(deepcopy(new_poly))
        elif len(self.cur_polygons[-1]["pts"]) == 0:
            self.cur_polygons[-1] = deepcopy(new_poly)
        else:
            self.cur_polygons.append(deepcopy(new_poly))

    def draw_polygons(self):
        [d.remove() for d in self.drawn if self.drawn and d]
        self.drawn = []
        fill_alpha = self.fill
        clrs_lines = ["#fc0303","#fc8003","#fcce03","#e8f2a7","#3dfc03","#ffffff","#03f4fc","#0384fc","#ff42bd","#000000"] 
        clrs_fill = ["#ffffff","#fc0303","#cafc03","#1403fc"]
        marker_types = ["x","o","d","s"]
        #You can add more Hexadecimal color codes if necessary. There are clrs_lines * clrs_fill amount of colors
        #messy code that loops through clrs_lines, if done, starts over and moves clrs_fill over by one,....
        for poly in self.cur_polygons:
            L_idx = poly["idx"]
            for i in range(len(clrs_fill)):
                if 10*(1+i) > L_idx >= 10*(i): 
                        F_idx = i
            if len(poly["pts"]) > 0:
                if L_idx >= len(clrs_lines):
                    L_idx = L_idx - len(clrs_lines)*(F_idx+1)
                polygons = np.array(poly["pts"])
                self.drawn.extend(self.ax.fill(polygons[:, 0], polygons[:, 1], clrs_fill[F_idx], alpha=fill_alpha))
                self.drawn.extend(self.ax.plot(polygons[:, 0], polygons[:, 1], color=clrs_lines[L_idx], marker=marker_types[F_idx]))
        self.ax.figure.canvas.draw_idle()

    ###########################################
    ## Drawing
    ###########################################

    def update(self):
        self.ax.figure.canvas.draw_idle()

    def save_label_images(self, plot=1):
        filenames = self.filenames
        polygons = self.polygons
        for i, fn in enumerate(filenames):
            poly = polygons[i]
            if i <= self.im_idx:
                print("saving labels for " + str(len(poly)) + " objects:" + fn)
                im = self.read_image(fn)
                labeled = self.gen_index_image(im, poly)
                if len(poly) == 0:
                    x,y,d = np.shape(im)
                    labeled = np.zeros([x,y], dtype = np.uint8)
                self.save_labeled(fn, labeled)
                print('labels saved')

    def save_current_label_images(self, plot=1):
        fn = self.filenames[self.cur_idx]
        if len(self.cur_polygons) == 0:
            pass
        print("saving labels for " + str(len(self.cur_polygons)) + " objects: " + fn)
        im = self.read_image(fn)
        labeled = self.gen_index_image(im, self.cur_polygons)
        if plot:
            plt.imshow(labeled)
            plt.title('labeled')
            plt.show()
        self.save_labeled(fn, labeled)
        print('labels saved')

    def gen_index_image(self, im, poly): #Makes a new np.array and fills the areas that were annotated with a polygon with 1, the rest with 0
        shape = np.array(im).shape[0:2] #Takes original image shape, [0:2] because shape returns (height,width,layers)
        label_im = np.zeros(shape, np.uint8)
        for labels in poly:
            lab_idx, poly = labels["idx"], labels["pts"]
            if len(poly) == 1:
                poly_coor = np.round(poly)[:, ::-1]
                mask = disk(poly_coor[0, 0], poly_coor[0, 1], 10)
                label_im[mask] = lab_idx + 1
            if len(poly) > 2:
                poly_coor = np.round(poly)[:, ::-1]
                mask = polygon2mask(shape, poly_coor)
                label_im[mask] = lab_idx + 1
        return label_im

    def save_labeled(self, fn, label_im):
        # Add switch if fn is tiff / H5
        save_labeled(fn, label_im)


###########################################
#     Helper Functions 
###########################################
import h5py

def save_labeled(fn, label_im):
    if ".jpg" in fn.lower() or ".tif" in fn.lower():
        fn_out = fn.replace('.jpg', '_label.jpg')
        label_im = label_im*255 #outgoing image is an array with 0 where the user didnt draw polygons and 1 where they did,
                                #since image tools like the built-in one in everyones pc needs values between 0 and 255, having
                                #a value of 1 would be practically invisible for us.
        io.imsave(fn_out,label_im)
    if ".png" in fn.lower():
        fn_out = fn.replace('.png', '_label.png')
        label_im = label_im*255
        io.imsave(fn_out,label_im)
    if ".h5" in fn.lower():
        add_dataset_hypercube(fn, label_im, dataset="labels")
        if fn.find("_labeled_hypercube") == -1:
            os.rename(fn, fn.replace("_hypercube", "_labeled_hypercube"))

def read_H5(fn, dataset="mask_data"):
    with h5py.File(fn, 'r') as fh:
        if dataset not in fh.keys():
            return None
        data = np.array(fh[dataset][:])
    if data.dtype == np.uint8:
        pass
    else:
        if data.max() <= 1:  data = (data * 255).astype(np.uint8)
    return data

def read_H5_hypercube(fn, dataset="data", all = []):
    with h5py.File(fn, 'r') as fh:
        if dataset not in fh['hypercube'].keys():  # ammended to read the hypercube format
            return None
        if all == []:
            data = np.array(fh['hypercube'][dataset][:])
        else:
            data = np.array(fh['hypercube'][dataset][:,:,all])
    if data.dtype == np.uint8:
        pass
    else:
        if data.max() <= 1:  data = (data * 255).astype(np.uint8)
    return data

def read_image(self, filename):
    im = cv2.imread(filename)
    return im

def add_dataset(fn, data, dataset="labels"):
    with h5py.File(fn, 'r+') as fh:
        if dataset in fh.keys():
            del fh[dataset]
            fh[dataset] = data
        else:
            fh.create_dataset(dataset, data=data, compression="lzf")
    fn.close()

def add_dataset_hypercube(fn, data, dataset="labels"):
    with h5py.File(fn, 'r+') as fh:
        if dataset in fh['hypercube'].keys():
            del fh['hypercube'][dataset]
            fh['hypercube'][dataset] = data
        else:
            fh['hypercube'].create_dataset(dataset, data=data, compression="lzf")
    fh.close()

def resize_label(fn):
    lbl = read_H5(fn, dataset="labels")
    if lbl is None:
        return
    msk = read_H5(fn, dataset="mask_data")
    lbl = cv2.resize(lbl, msk.shape[::-1])
    add_dataset(fn, lbl, dataset="labels")

def resize_label_hypercube(fn):
    lbl = read_H5_hypercube(fn, dataset="labels")
    if lbl is None:
        return
    msk = read_H5_hypercube(fn, dataset="data")
    lbl = cv2.resize(lbl, msk.shape[::-1])
    add_dataset_hypercube(fn, lbl, dataset="labels")

#Run the script
if __name__ == "__main__":
    an = Annotate(exclude_labeled=True, run=True, bands=[0,1,2,3,4])
    plt.show()
    an.save_label_images()
    an.writetoprefs()