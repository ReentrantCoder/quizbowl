import math
from string import split

from numpy.core.fromnumeric import reshape

from sklearn.dummy import DummyClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors.kde import KernelDensity

class User:
    def __init__(self, userId):
        self.userId = userId
        self.test = []
        self.train = []

class OriginalModel:
    def __init__(self, user, featurizer):
        self.user = user
        self.featurizer = featurizer

        # P(A | Q)
        y = featurizer.getY(user.train) 
        if True in y and False in y:    
            self.probAGivenQ = LogisticRegression()
            self.probAGivenQ.fit(featurizer.getX(user.train), y)
        elif len(y) > 0:
            self.probAGivenQ = DummyClassifier(strategy='constant', constant=float(y[0]))
        else:
            self.probAGivenQ = DummyClassifier(strategy='constant', constant=0)
            

        # P(N | A = Correct)
        NC = featurizer.getN(user.train, True)
        NC = reshape(NC, (len(NC), 1))
        
        if len(NC) > 0:
            self.probNGivenCorrect =  KernelDensity(kernel='gaussian', bandwidth=1)
            self.probNGivenCorrect.fit(NC)
        else:
            self.probNGivenCorrect = DummyClassifier(strategy='constant', constant = 1)
            
        # P(N | A = Incorrect)
        NI = featurizer.getN(user.train, False)
        NI = reshape(NI, (len(NI), 1))
        
        if len(NI) > 0:
            self.probNGivenIncorrect = KernelDensity(kernel='gaussian', bandwidth=1)
            self.probNGivenIncorrect.fit(NI)
        else:
            self.probNGivenIncorrect = DummyClassifier(strategy='constant', constant = 0)

    def getExpectedPosition(self, query):
        # how to make use of yp and xp?
        (y, yp) = self.argmax(query, True)
        (x, xp) = self.argmax(query, False)

        if y < x:
            return y

        return -x

    def argmax(self, query, boolA):
        optProb = float("-inf")
        optN = None
        
        maxN = len(split(query.questionText))
        
        for n in xrange(0, maxN):
            query.position = n
            
            prob = self.probAGivenQCN(boolA, query)
            if prob > optProb:
                optProb = prob
                optN = n
                
        return (optN, optProb)

    def probAGivenQCN(self, boolA, query):
        # P( A | Q, C, N) \propto P(Q | A) P(N | A)

        x = self.featurizer.getSingleX(query)
        
        lrProbWrongRight = self.probAGivenQ.predict_proba(x)
        probQGivenIncorrect = lrProbWrongRight[0][0]
        probQGivenCorrect = lrProbWrongRight[0][1]

        kde = self.probNGivenCorrect
        probQGivenA = probQGivenCorrect
        if not boolA:
            kde = self.probNGivenIncorrect
            probQGivenA = probQGivenIncorrect

        probNGivenA = math.exp(kde.score(self.featurizer.getSingleN(query, boolA)))
        
        return probQGivenA * probNGivenA

    def logisticFunc(self, x):
        if x < -10:
            return 0
        
        if x > 10:
            return 1.0

        return 1.0 / (1.0 + math.exp(-x))
