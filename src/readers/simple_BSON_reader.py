import bson


class SimpleBSONReader:

    def __init__(self, filename):
        self.fileName = filename

    def read(self):
        bson_file = open(self.fileName, 'rb')

        return bson.loads(bson_file.read())
