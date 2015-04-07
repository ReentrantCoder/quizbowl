from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.neighbors.kde import KernelDensity
from sklearn.dummy.DummyClassifier import DummyClassifier
import math

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
            self.probAGivenQ = SGDClassifier()
            self.probAGivenQ.fit(featurizer.getX(user.train), featurizer.getY(user.train))
        else:
            self.probAGivenQ = DummyClassifier(strategy='constant', constant=float(y[0]))

        # P(N | A = Correct)
        self.probNGivenCorrect =  KernelDensity(kernel='gaussian', bandwidth=1)
        self.probNGivenCorrect.fit(featurizer.getN(user.train, True))
        
        # P(N | A = Incorrect)
        self.probNGivenIncorrect = KernelDensity(kernel='gaussian', bandwidth=1)
        self.probNGivenIncorrect.fit(featurizer.getN(user.train, False))

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
        
        for n in xrange(0, 100):
            query.position = n
            
            prob = self.probAGivenQCN(boolA, query)
            if prob > optProb:
                optProb = prob
                optN = n
                
        return (optN, optProb)

    def probAGivenQCN(self, boolA, query):
        # P( A | Q, C, N) \propto P(Q | A) P(N | A)

        x = self.featurizer.getSingleX(query)
        
        probQGivenIncorrect = self.logisticFunc( -(x * self.probAGivenQ.coef_ + self.probAGivenQ.intercept_) )
        probQGivenCorrect = 1.0 - probQGivenIncorrect

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
