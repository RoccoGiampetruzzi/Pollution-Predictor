import numpy as np
from sklearn.decomposition import PCA


class PM_HIGH_PREDICTOR:
    def __init__(self, K):
        self.mean = None
        self.std = None
        self.pca = None
        self.K = K
        self.centroids = None
        self.labels = None
    
    def normalize(self, X, training = False, pca = False):
        
        """
        Inputs:
        - X (numpy array): Array containing the input data.
        """
        
        if training:                                # mean, std and pca values can only be set during training
            self.mean = np.mean(X, axis = 0)        # Mean over the rows of the input array
            self.std = np.std(X, axis = 0)          # STD over the rows of the input array

            X = (X - self.mean)/self.std
        
            if pca:
                self.pca = PCA(n_components=6)
                X = self.pca.fit_transform(X)
            return X
        
        else:
            try:
                X = (X - self.mean)/self.std
                if self.pca is not None:
                    X = self.pca.fit_transform(X)
            except:
                raise Exception('')
            return X
        
        
    def fit(self, X_train, y_train, n_iterations = 10, tolerance = 1e-4, verbose=False, pca = True):
        
        """
        Inputs:
        - X (np.array): training data
        - verbose: verbosity parameter (set to True to see distance and # of iterations)
        Outputs:
        - None
        """
        
        if len(X_train.shape) == 1:
            raise Exception('You need at least two vectors to train K-Means')
        
        ### INPUT DATA INITIALIZATION ###
        X = self.normalize(X_train, training=True, pca=pca)                                                           
        
        best_centroids = np.zeros((self.K, X.shape[1]))
        best_inertia = 100000
        best_centroid_assignment = np.zeros(X.shape[1])
        
        for iteration in range(n_iterations):
            
            ### RANDOM CENTROID INITIALIZATION ###
            self.centroids = np.random.normal(loc=0, scale= 1, size = (self.K, X.shape[1]))     
            
            centroid_distance = 100
            n_iterations = 0
            
            
            #### ITERATIVE STEP ####
            while centroid_distance > tolerance:
                
                # Computing distance between each datapoint and each centroid (returns array of shape n_datapoints x n_centroids)
                point_centroid_distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis = 2)
                
                # Assigning each datapoint to closest centroid (returns array of shape n_datapoints x 1, with entries in [0, K])
                centroid_assignment = np.argmin(point_centroid_distances, axis = 1)

                
                new_centroids = []
                for i in range(self.K):
                    if np.bincount(centroid_assignment, minlength=self.K + 1)[i]:
                        
                        # Computing new centroids by doing mean of assigned datapoints
                        new_centroids.append(np.mean(X[centroid_assignment == i], axis = 0))
                    
                    else:
                        new_centroids.append(self.centroids[i])
                
                new_centroids = np.array(new_centroids)
                centroid_distance = np.linalg.norm(self.centroids - new_centroids)
                self.centroids = new_centroids
                n_iterations += 1
                # if verbose:
                #     print(f'ITERATION {n_iterations}. DISTANCE: {centroid_distance}')
                #     print(np.bincount(centroid_assignment))
            
            current_intertia = np.sum(point_centroid_distances[np.arange(point_centroid_distances.shape[0]), centroid_assignment])   
            if verbose:
                print(f'Iteration {iteration}. Inertia: {current_intertia:.2f}')
            if current_intertia < best_inertia:
                best_inertia = current_intertia
                best_centroids = new_centroids
                best_centroid_assignment = centroid_assignment
        
        
        self.centroids = best_centroids
        self.labels = np.zeros(self.K)
        for i in range(self.K):
            if len(np.bincount(y_train[best_centroid_assignment == i].astype(int))):
                self.labels[i] = np.argmax(np.bincount(y_train[best_centroid_assignment == i].astype(int)))
        if verbose:
            print('Training Complete')
            
    def predict(self, X_test):
        if self.centroids is None:
            raise Exception('The model has not been trained')
        
        if len(X_test.shape) == 1:
            X_test = X_test[np.newaxis, :]
        
        X = self.normalize(X_test, training=False)
        
        point_centroid_distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis = 2)
        point_assignment = np.argmin(point_centroid_distances, axis = 1)
        return self.labels[point_assignment]
        
        
    def score(self, X, y):
        y_pred = self.predict(X)
        n_labels = max(int(np.max(self.labels)) + 1, int(np.max(y))+1)
        confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
        
        # Counting entries of Confusion Matrix
        for true_label, pred_label in zip(y, y_pred):
            confusion_matrix[int(true_label), int(pred_label)] += 1
            
        # Computing accuracy
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        
        return accuracy, confusion_matrix
    