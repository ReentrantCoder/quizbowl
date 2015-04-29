from fileFormats import TrainFormat, TestFormat, QuestionFormat
from data import Dataset

if __name__ == '__main__':
    dataset = Dataset(TrainFormat(), TestFormat(), QuestionFormat())
    train, test = dataset.getTrainingTest("data/train.csv", "data/test.csv", "data/questions.csv", -1)    

    D = {}

    for t in train:
        if not t.questionCategory in D:
            D[t.questionCategory] = []

        count = 0

        words = t.questionText[0:t.position].split()
        for (prev, next) in zip(words, words[1:]):
            if prev.endswith(",") or prev.endswith(".") or prev.endswith(";") or prev.lower() in ["and", "or"]:
                count += 1

        D[t.questionCategory].append(( (t.position) / float(len(t.questionText)), count, ))
        
    for (category, s) in D.items():
        for (pos, count) in s[0:500]:
            print("%s,%f,%d" % (category, pos, count))
