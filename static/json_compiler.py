import json
import os

class json_handler():
    def __init__(self, parent=None):
        super().__init__()
        self.filename = "./paths/data_test.json"

    def transfer(self,data):
        if data == []:
            print("nothing to compile")
            return 
        jsonString = json.dumps(data)
        jsonFile = open(self.filename, "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    def unpack(self):
        file = open(self.filename, "r")
        temp = file.read()
        recording = json.loads(temp)
        file.close()
        return recording
    
    def presets_json(self):
        print("work on it.")
