import numpy as np
from sklearn.preprocessing import StandardScaler



def load_data(filename: str):
    """
    Load data from /data folder.

    Parameters:
    - filename: string, name of the file to load

    Returns:
    - X: numpy array of shape (N, d), N unlabeled sample points in a d-dimensional space
    - Y: numpy array of shape (N,), N labels corresponding to the sample points
    """
    if filename == 'winequality-red':

        data = np.loadtxt('data/winequality-red.csv', delimiter=';', skiprows=1)
        X = data[:, :-1]
        Y = data[:, -1]
        return X, Y
    elif filename == 'winequality-white':
        data = np.loadtxt('data/winequality-white.csv', delimiter=';', skiprows=1)
        X = data[:, :-1]
        Y = data[:, -1]
        return X, Y
    elif filename == 'california_housing':
        data = np.loadtxt('data/california_housing.csv', delimiter=',', skiprows=1)
        X = data[:, :-1]
        Y = data[:, -1]
        return X, Y
    
    # Preprocess data
    # Normalize data to have zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Return data
    return X, Y