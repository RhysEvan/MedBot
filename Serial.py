from sys import platform
import time
import serial
import serial.tools.list_ports as port_list

class serial_bridge:
    def __init__(self, port = None):
    
        if port is None:
            #self.device = self.connect_to_ports("")
            #print(self.device)
            print("no current arduino present")
            self.device = None
        else:
            self.device = serial.Serial(port, 155200, timeout=0.1)
    
    def open_bridge(self):

        # Open grbl serial port        
        if platform == "linux" or platform == "linux2":
            raise Exception( "Not Linux compatible yet" )
            #rasp /dev/ttyUSB0. For Pc COM

        elif platform == "darwin":  ##OS X
            raise Exception( "Not Mac compatible yet" )
            
        elif platform == "win32":
            self.s = self.connect_to_ports_win()

        if self.s is None:   return
        # Open g-code file
        # Wake up grbl
        self.s.write(b"\r\n\r\n")
        time.sleep(2)   # Wait for grbl to initialize
        self.s.flushInput()  # Flush startup text in serial input

    def connect_to_ports_win(self, find_name):
        all_ports = list(port_list.comports())
        pos_ports = [p.device for p in all_ports  if "Arduino" in p.description]
        [print(p.description) for p in all_ports]

        if pos_ports == []:
            all_ports = list(port_list.comports())
            ard = all_ports[0].device
            ard = serial.Serial(port, 115200, timeout=0.1, write_timeout=0.1,inter_byte_timeout=0.1)
            return ard

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
    
    def connect_to_ports_linux(self):
        all_ports = list(port_list.comports())
        ard = all_ports[0].device
        ard = serial.Serial(ard, 115200, timeout=0.1, write_timeout=0.1,inter_byte_timeout=0.1)
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
    
    def send_com(self, input):
        if self.device is None:
            return   
        self.device.write(bytearray(str(input)+"\r\n","utf-8"))
    
    def home(self):
        print("homing")
        self.com.send_com("$X")
        self.com.send_com("$F 100")
        self.com.send_com("$HX")
        #self.com.send_com("$HY")