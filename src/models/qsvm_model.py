from sklearn.svm import SVC

class QSVM:
    def __init__(self, quantum_kernel):
        self.kernel = quantum_kernel
        self.model = SVC(kernel='precomputed')

    def fit(self, X_train, y_train):
        K_train = self.kernel(X_train, X_train)
        self.model.fit(K_train, y_train)

    def predict(self, X_test, X_train):
        K_test = self.kernel(X_test, X_train)
        return self.model.predict(K_test)
    
    def accuracy(self, X_test, y_test, X_train):
        y_pred = self.predict(X_test, X_train)
        return (y_pred == y_test).mean()