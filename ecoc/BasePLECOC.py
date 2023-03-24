# from sklearn.svm import libsvm
from sklearn.svm import SVC as libsvm


class BasePLECOC:
    def __init__(self, estimator=libsvm, **params):
        self.estimator = estimator
        self.params = params

    def fit(self, train_data, tarin_labels):
        pass

    def predict(self, test_data):
        pass
