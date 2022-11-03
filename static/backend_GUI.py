import copy

class backend():
    def __init__(self, main):
        limits = copy.copy(main.graph.limits)
        limits.remove([])
        print(len(limits))
        ## visual locations of the graph when initializing ##
        self.absolute_a = "0"
        main.text_aabs.setText("0")
        self.absolute_b = "100"
        main.text_babs.setText("100")
        self.absolute_c = "270"
        main.text_cabs.setText("270")
        self.absolute_d = "80"
        main.text_dabs.setText("80")
        self.absolute_e = "40"
        main.text_eabs.setText("40")
        ## initial values for the recording list of xyz values ##
        self.x_loc = "0"
        self.y_loc = "0"
        self.z_loc = "0"
        self.alfa_loc = "0"
        self.beta_loc = "0"
        self.gamma_loc = "0"
