import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import copy
from sklearn.ensemble import AdaBoostRegressor
from scipy.stats import pearsonr 
import random
import os.path
import sys
import os.path
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
print(parent_dir)
from process_data import load_data

sys.path.remove(parent_dir) 

seed = 41
np.random.seed(seed)  
random.seed(seed)

class Active_Learner:

    def __init__(self, X: np.ndarray, y: np.ndarray, minimum:int, maximum: int, query_strategy: str, model:object, number_of_models:int) -> None:
        np.random.seed(seed)  
        random.seed(seed)
        self.X = X
        self.y = y
        
        self.query_strategy = query_strategy

        self.model = model
        self.number_of_models = number_of_models

        numY0 = len(self.y)
        self.minN = min(minimum, self.X.shape[1] + 1)  # mininum number of training samples
        self.maxN = min(maximum, max(minimum, int(0.1 * numY0)))  # maximum number of training samples

        # Use train_test_split to get pool and test indices
        self.pool_indices, self.test_indices = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(self.X, self.y))
        #get pool and test set 
        self.X_pool = self.X[self.pool_indices]
        self.y_pool = self.y[self.pool_indices]
        self.X_test = self.X[self.test_indices]
        self.y_test = self.y[self.test_indices]

        self.rmse_vals = []
        self.cc_vals = []
        self.rmse_vals_base = []
        self.cc_vals_base = []
        self.mae_vals = []

    #add values that will be deleted from pool to train and delete pool values

    def pool_delete(self, removed_indices:np.ndarray)->None:
        """
        This method removes selected sample from pool given index and adds to training.
        Args:
            removed_indices: The index of sample to be removed.
        """
        self.X_train = np.append(self.X_train, self.X_pool[removed_indices], axis=0)
        self.y_train = np.append(self.y_train, self.y_pool[removed_indices])
        self.train_indices = np.append(self.train_indices, self.pool_indices[removed_indices])
        self.X_pool = np.delete(self.X_pool, removed_indices, axis = 0)
        self.y_pool = np.delete(self.y_pool, removed_indices)
        self.pool_indices = np.delete(self.pool_indices, removed_indices)
    
    def set_training_data(self)->None:
        """
        This method sets the inital training set given by minN hyperparamter.
        Some number of samples are labeled to be considerd as training data.
        """

        removed_indices = random.sample(range(self.minN), self.minN)
        self.X_train = self.X_pool[removed_indices]
        self.y_train = self.y_pool[removed_indices]
        self.train_indices = self.pool_indices[removed_indices]
        self.X_pool = np.delete(self.X_pool, removed_indices, axis = 0)
        self.y_pool = np.delete(self.y_pool, removed_indices)
        self.pool_indices = np.delete(self.pool_indices, removed_indices)

    def query(self)->None:
        """
        This method performs a query strategy given the query_strategy type.
        """
        for N in range(self.minN, self.maxN):

            if self.query_strategy == 'random':
                self.BL()

            elif self.query_strategy == 'committee':
                self.QBC()

            elif self.query_strategy == 'weights':
                self.QBC_weights()

            elif self.query_strategy == 'boost':
                self.QBC_boosting()
            
            elif self.query_strategy == 'average':
                self.QBC_average()

            elif self.query_strategy == 'stacking':
                self.QBC_stacking()

            elif self.query_strategy == 'weight average':
                self.QBC_weights_average()

        
    def BL(self)->None:
        """
        This method performs a random query strategy where instance in pool is randomly chosen.
        """
        np.random.seed(seed)  
        random.seed(seed)
        # Randomly select samples from the available pool
        random_id = random.choice(range(len(self.y_pool)))
        random_arr = np.array([random_id])
        self.pool_delete(removed_indices = random_arr)
        self.model.fit(self.X_train, self.y_train)

        predicted = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted)
        self.rmse_vals.append(mse)
        cc, _ = pearsonr(self.y_test, predicted)
        self.cc_vals.append(cc)
        mae = mean_absolute_error(self.y_test, predicted)
        self.mae_vals.append(mae)

    #this approach is similar to query by bagging used in classification [3]. It was argued that this approach should lead to better performance than [7]
    #pseudo bagging like in classifiers
    def QBC(self) -> None:
        """
        This method performs a QBC query strategy specified by RayChaudhuri and Hamey.
        Models are trainined on random subsets. Instance with maximal variance chosen to query on.
        New model trained and evaluated with additional instance.
        """
        np.random.seed(seed)  
        random.seed(seed)
        # Train multiple models on slightly different subsets
        models = [copy.deepcopy(self.model) for _ in range(self.number_of_models)]  

        for i in range(self.number_of_models):
            subset_size = np.random.randint(2, len(self.X_train) + 1)
            subset_indices = np.random.choice(len(self.y_train), subset_size, replace=True)  # Use replace=True as paper mentioned we are doing pseudo bagging
            X_subset = self.X_train[subset_indices]
            y_subset = self.y_train[subset_indices]

            # Train the model on the subset
            models[i].fit(X_subset, y_subset)

        # Query the instance with the highest variance of predictions
        all_predictions = np.array([model.predict(self.X_pool) for model in models])
        variance = np.var(all_predictions, axis=0)
        index = np.argmax(variance)

        # Delete the queried instance from the pool
        self.pool_delete(removed_indices=np.array([index]))

        self.model.fit(self.X_train, self.y_train)

        # Eval model
        predicted = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted)
        self.rmse_vals.append(mse)
        cc, _ = pearsonr(self.y_test, predicted)
        self.cc_vals.append(cc)
        mae = mean_absolute_error(self.y_test, predicted)
        self.mae_vals.append(mae)

    #adaboost for active regression
    def QBC_boosting(self) -> None:
        """
        This method performs a QBC boosting strategy as an EXTENSION.
        Models are trainined on random subsets. Instance with maximal variance chosen to query on.
        New model is boosting model trained and evaluated with additional instance.
        """
        np.random.seed(seed)  
        random.seed(seed)
        base_learner = Ridge(alpha=0.1)

        boosting_model = AdaBoostRegressor(estimator=base_learner, n_estimators=self.number_of_models)

        # Initialize an array to store the predictions of each base learner
        all_predictions = np.zeros((self.number_of_models, len(self.y_pool)))

        for i in range(self.number_of_models):
            subset_size = np.random.randint(2, len(self.X_train) + 1)
            subset_indices = np.random.choice(len(self.y_train), subset_size, replace=True)  # Use replace=False to ensure unique indices
            X_subset = self.X_train[subset_indices]
            y_subset = self.y_train[subset_indices]

            # Train the base learner on the subset
            base_learner.fit(X_subset, y_subset)

            # Store the predictions of the base learner
            all_predictions[i] = base_learner.predict(self.X_pool)

        # Query the instance with the highest variance of predictions
        variance = np.var(all_predictions, axis=0)
        index = np.argmax(variance)

        # Delete the queried instance from the pool
        self.pool_delete(removed_indices=np.array([index]))

        # Retrain the boosting model 
        boosting_model.fit(self.X_train, self.y_train)

        # Eval boosting 
        predicted = boosting_model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted)
        self.rmse_vals.append(mse)
        cc, _ = pearsonr(self.y_test, predicted)
        self.cc_vals.append(cc)
        mae = mean_absolute_error(self.y_test, predicted)
        self.mae_vals.append(mae)

    #active regression models trained on same data but differnet 'weights' i.e. hyperparamters of alpha. 
    def QBC_weights(self) -> None:
        """
        This method performs a QBC strategy specified by Krogh and Vedelsby.
        Models are trainined on same data. Unqiueness comes from randomization of weights of models.
        As this is an adaptation from original Neural Network arc, ridge regressor alpha is randomized.
        Instance with maximal variance chosen to query on.
        New model is trained and evaluated with additional instance.
        """
        np.random.seed(seed)  
        random.seed(seed)
        # Train multiple models on slightly different subsets with randomized alphas
        models = [copy.deepcopy(self.model) for _ in range(self.number_of_models)] 
        alpha_values = [0.01, 0.1, 1]
        for i in range(self.number_of_models):

            # Train the model on the subset with a randomized alpha value
            random_alpha = np.random.choice(alpha_values)
            model = copy.deepcopy(self.model)
            model.set_params(alpha=random_alpha)
            model.fit(self.X_train, self.y_train)
            models[i] = model

        # Query the instance with the highest variance of predictions
        all_predictions = np.array([model.predict(self.X_pool) for model in models])
        variance = np.var(all_predictions, axis=0)
        index = np.argmax(variance)

        # Delete the queried instance from the pool
        self.pool_delete(removed_indices=np.array([index]))

        # Retrain the base model
        self.model.fit(self.X_train, self.y_train)

        predicted = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted)
        self.rmse_vals.append(mse)
        cc, _ = pearsonr(self.y_test, predicted)
        self.cc_vals.append(cc)
        mae = mean_absolute_error(self.y_test, predicted)
        self.mae_vals.append(mae)

    #active regression with stacking
    def QBC_stacking(self) -> None:
        """
        This method performs QBC strategy as an EXTENSION.
        Models are trained on random subsets. Each base model then makes predictions.
        Base predictions used as traininig for metamodel regressor. 
        Instance with maximal variance chosen to query on.
        New model is trained and evaluated with additional instance.
        """
        np.random.seed(seed)  
        random.seed(seed)
        # Train multiple models on different subsets
        base_models = [copy.deepcopy(self.model) for _ in range(self.number_of_models)]

        # Train base models on different subsets
        for i in range(self.number_of_models):
            subset_size = np.random.randint(2, len(self.X_train) + 1)
            subset_indices = np.random.choice(len(self.y_train), subset_size, replace=True)
            X_subset = self.X_train[subset_indices]
            y_subset = self.y_train[subset_indices]

            # Train on subset
            base_models[i].fit(X_subset, y_subset)

        # Use a meta-model to combine predictions of base models
        meta_model = copy.deepcopy(Ridge(alpha=0.01))

        meta_predictions = np.array([model.predict(self.X_pool) for model in base_models]).T
        meta_model.fit(meta_predictions, self.y_pool)

        # Query instance with the maximal variance in the meta-model prediction
        meta_predictions_test = np.array([model.predict(self.X_test) for model in base_models]).T
        #predictions_pool = np.array([model.predict(self.X_pool) for model in base_models]).T
        predictions_pool = np.array([model.predict(self.X_pool) for model in base_models])
        variance = np.var(predictions_pool, axis=1)
        index = np.argmax(variance)

        # Delete the queried instance from the pool
        self.pool_delete(removed_indices=np.array([index]))

        # Evaluate  and store 
        predicted = meta_model.predict(meta_predictions_test)
        mse = mean_squared_error(self.y_test, predicted)
        self.rmse_vals.append(mse)
        cc, _ = pearsonr(self.y_test, predicted)
        self.cc_vals.append(cc)
        mae = mean_absolute_error(self.y_test, predicted)
        self.mae_vals.append(mae)
 
    def QBC_average(self) -> None:
        """
        This method performs QBC strategy as an EXTENSION.
        Models are trained on random subsets. Each base model then makes predictions.
        Instance with maximal variance chosen to query on.
        Average of ensembles is evaluated.
        """
        np.random.seed(seed)  
        random.seed(seed)
        # Train multiple models on slightly different subsets
        models = [copy.deepcopy(self.model) for _ in range(self.number_of_models)]

        for i in range(self.number_of_models):
            subset_size = np.random.randint(2, len(self.X_train) + 1)
            subset_indices = np.random.choice(len(self.y_train), subset_size, replace=True)
            X_subset = self.X_train[subset_indices]
            y_subset = self.y_train[subset_indices]

            # Train the model on the subset
            models[i].fit(X_subset, y_subset)

        # Query instance with maximal variance 
        all_predictions = np.array([model.predict(self.X_pool) for model in models])
        ensemble_prediction = np.mean(all_predictions, axis=0)
        index = np.argmax(np.var(ensemble_prediction))

        # Delete query instance from the pool
        self.pool_delete(removed_indices=np.array([index]))

        # Evaluate and store
        predicted = np.mean([model.predict(self.X_test) for model in models], axis=0)
        mse = mean_squared_error(self.y_test, predicted)
        self.rmse_vals.append(mse)
        cc, _ = pearsonr(self.y_test, predicted)
        self.cc_vals.append(cc)
        mae = mean_absolute_error(self.y_test, predicted)
        self.mae_vals.append(mae)

    #active regression models trained on same data but differnet 'weights' i.e. hyperparamters of alpha. 
    def QBC_weights_average(self) -> None:
        """
        This method performs QBC strategy as an EXTENSION.
        Models are trained on random alpha. Each base model then makes predictions.
        Instance with maximal variance chosen to query on.
        Average of ensembles is evaluated.
        """
        np.random.seed(seed)  
        random.seed(seed)
        # Train multiple models on slightly different subsets with randomized alphas
        models = [copy.deepcopy(self.model) for _ in range(self.number_of_models)]
        alpha_values = [0.01, 0.1, 1]
        for i in range(self.number_of_models):

            # Train the model on the subset with a randomized alpha value
            random_alpha = np.random.choice(alpha_values)
            model = copy.deepcopy(self.model)
            model.set_params(alpha=random_alpha)
            model.fit(self.X_train, self.y_train)
            models[i] = model

        # Query instance with maximal variance 
        all_predictions = np.array([model.predict(self.X_pool) for model in models])
        ensemble_prediction = np.mean(all_predictions, axis=0)
        index = np.argmax(np.var(ensemble_prediction))

        # Delete the queried instance from the pool
        self.pool_delete(removed_indices=np.array([index]))

        # Retrain the base model
        self.model.fit(self.X_train, self.y_train)

        predicted = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predicted)
        self.rmse_vals.append(mse)
        cc, _ = pearsonr(self.y_test, predicted)
        self.cc_vals.append(cc)
        mae = mean_absolute_error(self.y_test, predicted)
        self.mae_vals.append(mae)


def average_performance_plots(filename:str, epochs:int, strategies: list,  minN:int, maxN:int, model, number_of_models:int)->None:

    X, y = load_data(filename=filename)
    metrics = {strategy: {'ccs': [], 'rmses': [], 'maes': []} for strategy in strategies}

    for strategy in strategies:
        for _ in range(epochs):
            learner = Active_Learner(X=X, y=y, minimum=minN, maximum=maxN, query_strategy=strategy, model=model, number_of_models=number_of_models)
            learner.set_training_data()
            learner.query()
            metrics[strategy]['ccs'].append(learner.cc_vals)
            metrics[strategy]['rmses'].append(learner.rmse_vals)
            metrics[strategy]['maes'].append(learner.mae_vals)

        for metric in metrics[strategy]:
            metrics[strategy][metric] = np.mean(np.array(metrics[strategy][metric]), axis=0)
    for metric_name, ylabel in [('ccs', 'CC'), ('rmses', 'RMSE'), ('maes', 'MAE')]:
        plt.figure(figsize=(10, 6))
        for strategy in strategies:
            plt.plot(range(learner.minN, learner.maxN), metrics[strategy][metric_name], marker='o', label=strategy)
        plt.xlabel('Labeled Instances')
        plt.ylabel(ylabel)
        plt.title(f'Red Wine Dataset - {ylabel} Comparison')
        plt.legend()
        plt.show()
    

def all_average_performance_plots(epochs: int, strategies: list, datasets: list, minN:int, maxN:int, model:object, number_of_models:int):
    for dataset in datasets:
        X, y = load_data(filename=dataset) 
        metrics = {strategy: {'ccs': [], 'rmses': [], 'maes': []} for strategy in strategies}

        for strategy in strategies:
            for _ in range(epochs):
                learner = Active_Learner(X=X, y=y, minimum=minN, maximum=maxN, query_strategy=strategy, model=model, number_of_models=number_of_models)
                learner.set_training_data()
                learner.query()
                metrics[strategy]['ccs'].append(learner.cc_vals)
                metrics[strategy]['rmses'].append(learner.rmse_vals)
                metrics[strategy]['maes'].append(learner.mae_vals)

            for metric in metrics[strategy]:
                metrics[strategy][metric] = np.mean(np.array(metrics[strategy][metric]), axis=0)

        for metric_name, ylabel in [('ccs', 'CC'), ('rmses', 'RMSE'), ('maes', 'MAE')]:
            plt.figure(figsize=(10, 6))
            for strategy in strategies:
                plt.plot(range(learner.minN, learner.maxN), metrics[strategy][metric_name], marker='o', label=strategy)
            plt.xlabel('Labeled Instances')
            plt.ylabel(ylabel)
            plt.title(f'{dataset} Dataset - {ylabel} Comparison')
            plt.legend()
            plt.show()


if __name__ == '__main__':
    all_average_performance_plots(epochs = 100, datasets = ['winequality-red', 'winequality-white', 'california_housing'], strategies = ['random', 'committee', 'weights'], minN = 20, maxN = 60,model= Ridge(alpha=0.01), number_of_models=5)
    all_average_performance_plots(epochs = 100, datasets = ['winequality-red', 'winequality-white', 'california_housing'], strategies = ['random', 'boost', 'stacking', 'average', 'weight average'], minN = 20, maxN = 60,model= Ridge(alpha=0.01), number_of_models=5)
    all_average_performance_plots(epochs = 100, datasets = ['winequality-red', 'winequality-white', 'california_housing'], strategies = ['random', 'committee', 'weights','boost', 'stacking', 'average', 'weight average'], minN = 20, maxN = 60,model= Ridge(alpha=0.01), number_of_models=5)
