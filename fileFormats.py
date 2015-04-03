import csv
from _collections import defaultdict

class FileFormat:
    def __init__(self):
        self.fieldnames = None
        self.skipHeader = False
        
    def deserialize(self, filePath):
        with open(filePath, 'rb') as csvFile:
            reader = csv.DictReader(csvFile, fieldnames=self.fieldnames)
            
            if self.skipHeader:
                next(reader)

            for row in reader:
                yield row
    
    def generatorToDict(self, generator):
        D = defaultdict(int)
        for entity in generator:
            D[entity["id"]] = entity
            
        return D;

    def serialize(self, entities, filePath):
        with open(filePath, 'wb') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=self.fieldnames)
            
            if not self.skipHeader:
                writer.writeheader()
            
            for entity in entities:
                writer.writerow(entity)

class QuestionFormat(FileFormat):
    def __init__(self):
        self.fieldnames=["id", "answer", "set", "category", "question", "blob"]
        self.skipHeader = False

class TrainFormat(FileFormat):
    def __init__(self):
        self.fieldnames=["id", "question", "user", "position", "answer"]
        self.skipHeader = True

class TestFormat(FileFormat):
    def __init__(self):
        self.fieldnames=["id", "question", "user"]
        self.skipHeader = True

class GuessFormat(FileFormat):
    def __init__(self):
        self.fieldnames=["id", "position"]
        self.skipHeader = False