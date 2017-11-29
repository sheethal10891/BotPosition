import numpy as np
import scipy
from sklearn.externals import  joblib
import pandas as pd
from hmmlearn import hmm
import math

def loadData(file_name='',data_type='float'):
    data=np.loadtxt('./data/'+file_name, delimiter=',',dtype=data_type)
    return data

#XRAW=loadData('Label.csv')
Obs=loadData('Observations.csv')
size = 500
Obs_aug = Obs[:size,:]
lengths_arr = np.array(size * [1000])
print(lengths_arr.shape)
Obs_row = Obs_aug.flatten().reshape(-1,1)
print(Obs_row.shape)

n_states = 36
clf = hmm.GaussianHMM(n_components=n_states, n_iter=1)
clf.fit(Obs_row, lengths = lengths_arr)

from sklearn.externals import joblib
joblib.dump(clf, "hmm_model.pkl")