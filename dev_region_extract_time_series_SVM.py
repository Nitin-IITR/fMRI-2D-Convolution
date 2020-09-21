import pandas as pd
import glob
import matplotlib.pyplot as plt
import scipy.signal as sps
import scipy.fftpack as spf
import numpy as np
import pywt
import scipy as sp


series =pd.concat([pd.DataFrame(time_series[i]) for i in range(30)],ignore_index=True) 

series1 =pd.concat([pd.DataFrame(time_series[i]) for i in range(30,60)],ignore_index=True) 

#
#############################

df = series.append(series1)
df= df.reset_index(drop=True)

    

###############################################

#X= df.iloc[:,[0., 1., 2., 3., 4., 5., 6., 7., 8.,36., 37., 38., 39., 40., 41., 42., 43., 44.]].values
#X= df.iloc[:,[40,41,42,43]].values

X= df.iloc[:,:].values
################################################
######### Nor 23.6 sec

#Y = np.concatenate((np.zeros(shape=456), np.ones(shape=456)))

#Y = np.concatenate((np.zeros(shape=855), np.ones(shape=855)))
Y = np.concatenate((np.zeros(shape=5040), np.ones(shape=5040)))

############################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)



acs=[]

for i in range(50):
    from sklearn.model_selection import train_test_split
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.25)
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_test)

    
#    from sklearn.metrics import confusion_matrix
#    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_test,Y_pred))


acs1 =np.mean(acs) 
acs1=acs1*100

acs2=np.std(acs)
acs2=acs2*100
