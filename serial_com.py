from queue import Empty
from sys import platform
import time
import serial
import serial.tools.list_ports as port_list

class serial_bridge:
    def __init__(self, port = None):
    
        if port is None:
            #self.device = self.bridge("")
            #print(self.device)
            print("no current arduino present")
            self.device = None
        else:
            self.device = serial.Serial(port, 155200, timeout=0.1)
        
        self.open_bridge()
    
    def bridge(self):

        # Open grbl serial port        
        if platform == "linux" or platform == "linux2":
            ports = self.check_ports()
            s = self.connect_to_ports_linux(ports)
            return s

        elif platform == "darwin":  ##OS X
            raise Exception( "Not Mac compatible yet" )
            
        elif platform == "win32":
            ports = self.check_ports()
            s = self.connect_to_ports_win(ports)
            return s

    def check_ports(self):
        all_ports = list(port_list.comports())
        pos_ports = [p.device for p in all_ports  if "Arduino" in p.description]
        [print(p.description) for p in all_ports]
        return pos_ports
    
    def connect_to_ports_linux(self):
        all_ports = list(port_list.comports())
        port = all_ports[0].device
        ard = serial.Serial(port, 115200, timeout=0.1, write_timeout=0.1,inter_byte_timeout=0.1)
        return ard

    def connect_to_ports_win(self, pos_ports, find_name):
        ## Search for Suitable Port
        print(pos_ports)
        for port in pos_ports: 
            print(".")
            try:      ard = serial.Serial(port, 115200, timeout=0.1, write_timeout=0.1, inter_byte_timeout=0.1)
            except:   continue
            #ard = serial.Serial(port, 115200, timeout=0.1, write_timeout=0.1, inter_byte_timeout=0.1)
            print("trying", port, "...", end="")
            response = self.read_info(ard)
            print(response, "...", end="")
            if response == find_name: 
                print("Port Found: ", port)
                break
            else:  
                ard.close()
                ard = None
        print("")

        return ard

    def read_info(self, ard):

        for _ in range(10): 
            response = ard.readline().decode("utf-8").split("\r")[0]
            if response == "":
                print(".",end="")
            if response == "Startup":
                print("Starting up device")
                time.sleep(.1)
                break         
        #ard.write(b"Info\r\n")
        Info = ard.readline().decode("utf-8").split("\r")[0]
        print("Device Info: "+ Info)  
        return Info

        
    def open_bridge(self):
        if self.device is None:   return
        # Open g-code file
        # Wake up grbl
        self.device.write(b"\r\n\r\n")
        time.sleep(2)   # Wait for grbl to initialize
        self.device.flushInput()  # Flush startup text in serial input

    def send_com(self, input):
        if self.device is None:
            print("no device")
            return   
        self.device.write(bytearray(input+"\r\n","utf-8"))
    
    def send_move(self, input):
        if self.device is None:
           print("no device")
           return   
        self.device.write(bytearray("$G1 "+str(input)+"\r\n","utf-8"))
    
    def home(self):
        print("homing")
        self.com.send_com("$X")
        self.com.send_com("$F 100")
        self.com.send_com("$HX")
        #self.com.send_com("$HY")