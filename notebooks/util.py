import numpy as np
import time

from datetime import timedelta
from sklearn import metrics

from sklearn.cross_validation import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split

# randomly reserve 10% of data and split the rest 75%-25% into train/test
def random_data_split(x_data, y_data):
    x_all, x_reserve, y_all, y_reserve = train_test_split(
        x_data.reshape(-1), 
        y_data[:,1].reshape(-1), 
        test_size=0.1, 
        random_state=np.random.RandomState())
    x_train, x_dev, y_train, y_dev = train_test_split(
        x_all, 
        y_all, 
        test_size=0.25, 
        random_state=np.random.RandomState())

    return x_reserve, y_reserve, x_train, y_train, x_dev, y_dev

# formats elapsed time into something readable
def elapsed_time(start, stop):
    label = 'hour(s)'
    time_str = str(timedelta(seconds = time.time()-start))
    if( time_str.startswith('0:')):
        time_str = time_str[2:]
        label = 'minute(s)'
        if( time_str.startswith('00:')):
            time_str = time_str[3:]
            label = 'second(s)'
    return 'Time elapsed: %s %s' % (time_str, label )    

# gridsearchcv with LR classifier
def knn_test(x_train_counts, y_train, x_dev_vectors , y_dev):
    #parameters = [{ 'n_neighbors':[1, 2, 3, 4, 5], 7, 10, 15, 20, 25, 30, 50] }]
    parameters = [{ 'n_neighbors':[1, 2, 3, 4, 5] }] 
    model = KNeighborsClassifier()
    clf = GridSearchCV(model, parameters)
    clf.fit(x_train_counts, y_train)
    predictions = clf.predict(x_dev_vectors)
    f1_score = metrics.f1_score( y_dev, predictions, average = 'weighted' )  
    
    return clf.best_score_ * 100.0, clf.best_params_['n_neighbors'], f1_score * 100.0

# gridsearchcv with knnclassifier
def lr_test(x_train_counts, y_train, x_dev_vectors , y_dev):
    parameters = [{'C': [1, 1000], 'penalty': ['l1']}]
    n=x_train_counts.shape[0]
    n_iterations = 10
    cv = ShuffleSplit(n, random_state=42)
    model = LogisticRegression(random_state=42)
    clf = GridSearchCV(model, parameters, cv=cv)
    clf.fit(x_train_counts, y_train)
    predictions = clf.predict(x_dev_vectors)
    f1_score = metrics.f1_score( y_dev, predictions, average = 'weighted' )  
    
    return clf.best_score_ * 100.0, clf.best_params_['C'], f1_score * 100.0