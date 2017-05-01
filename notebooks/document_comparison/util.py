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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

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
    time_str = str(timedelta(seconds = time.time()-start))
    if( time_str.startswith('0:')):
        time_str = time_str[2:]
        if( time_str.startswith('00:')):
            time_str = time_str[3:]
    return 'Time elapsed: %s' % time_str

def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
    """Pretty-print a timedelta, using the given format string."""
    since = since or time.time()
    until = until or time.time()
    delta_s = until - since
    hours, remainder = divmod(delta_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return fmt % (hours, minutes, seconds)


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


# find nearest neighbors of a given description using cosine similarity
def find_nn_cos( v, Dv, k=10):
    '''
    v: (d dimensional vector) representing the decription of interest 
    Dv: ( D x d matrix) decriptions
    k: number of neighbors to return
    
    returns (nns, ds):
    nns: (k-element vector of ints), 
         row indices of nearest neighbors, including the given description
    ds: (k-element vector of floats), 
        cosine similarity of each neighbor in nns
    '''
    def compute_cos_sim( wRow, v ):
        return (np.dot(v, wRow) / (np.linalg.norm(v) * np.linalg.norm(wRow)) * 1.0)

    # apply the function to all rows in Wv
    kwargs = {"v" : v }
    cs_arr = np.apply_along_axis(compute_cos_sim, 1, Dv, **kwargs )

    idx_arr = np.array(cs_arr).argsort()[::-1]
    cs_arr = cs_arr[idx_arr]
    
    # remove NaN
    d_idx = []
    for i in range(0, cs_arr.shape[0]):
        if np.isnan(cs_arr[i]):
            d_idx.append(i)
            
    cs_arr = np.delete(cs_arr, d_idx)
    idx_arr = np.delete(idx_arr, d_idx)
    
    return idx_arr[:k], cs_arr[:k]

def batch_generator(ids, batch_size, max_time):
    """Convert ids to data-matrix form."""
    # Clip to multiple of max_time for convenience
    clip_len = ((len(ids)-1) // batch_size) * batch_size
    input_w = ids[:clip_len]     # current word
    target_y = ids[1:clip_len+1]  # next word
    # Reshape so we can select columns
    input_w = input_w.reshape([batch_size,-1])
    target_y = target_y.reshape([batch_size,-1])

    # Yield batches
    for i in xrange(0, input_w.shape[1], max_time):
	yield input_w[:,i:i+max_time], target_y[:,i:i+max_time]

    '''
Function to compute precision, recall and f-score using knn and cosine similarity
Finds the k nearest neighbors ...
y_ is a vector of labels of length d
x_arr is a matrix of vectors of length d, width v
'''
def scoreVectorArray_CosSim(y_, x_arr):
    k=10
    y_pred=[]
    y_true=[]
    result=[]
    for i in range(0, x_arr.shape[0]):
        true_label = y_[i][0]
        vec = x_arr[i]
        r_idx, r_val = find_nn_cos(vec, x_arr, k)
        for j in range(0, r_idx.shape[0]):
            if r_val[j] != 1.0:
                y_true.append(true_label)
                y_pred.append(y_[r_idx[j]])
 
    
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print 'precision: %3.4f, recall: %3.4f, f_score: %3.4f' %( precision, recall, f_score)
    