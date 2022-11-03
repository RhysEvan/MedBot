import json

class json_handler():
    def __init__(self, parent=None):
        super().__init__()
        self.filename = "data.json"

    def transfer(self,data):
        print(data)
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
