"""
Approaches to try out:

1.'Benchmark' with standard features, feature selection and classifier. 
    Band powers and ratios
    Filter-based feature selection
    LDA/SVM/LR
2.Unsupervised pre-training
    Autoencoder pre-training
    Fine-tuning with labeled data
3.ConvNet
    Training on raw data
4.RNN/LSTM


Important points:
- Cross-validation! Avoid overfitting
- Features -> add ratios?
- 

"""

#MAKE A TABLE/PRESENTATION WITH EVERYTHING I WANT TO FIND OUT, AND THEN FILL IN 
#THE RESULTS
#USE IPYTHON NOTEBOOK?
# see http://nbviewer.ipython.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb
# 

import numpy as np
import pandas as pd

import scipy.io
import matplotlib.pylab as plt

from sklearn import decomposition, mixture, svm, linear_model
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.lda import LDA

#import SCIKIT!!!!!!!!!!!!

def loadData(filepath):
    """Load preprocessed data contained in a .mat file.
    
    Inputs
        filepath: Absolute path to the folder containing all the .mat files of EEG data
        
    Outputs
        eegData: pandas dataset of features
        
    TODO: make it possible to load more than one subject/session at the time
    TODO: return an error if size of data is strange
    """
    
    data = scipy.io.loadmat(filepath)
    
    # 'data' should contain fields 'CLASS', 'FEAT' and 'feat_names'

    feats = data['FEAT']
    labels = data['CLASS']
    featNames = data['feat_names']
    
    # List comprehension for cleaning the chList
    featNames_clean = [str(''.join(letter)) for letter_array in featNames[0] for letter in letter_array]

    assert (feats.shape[1] == featNames.shape[1]), 'The number of features is not unanimous between feats and featNames.'    
    
    # Make a pandas dataframe
    eegFeats = pd.DataFrame(data = feats, columns = featNames_clean)
    eegFeats['label'] = pd.Series(data = np.reshape(labels,(-1,)))

    return eegFeats
    
    
if __name__ == '__main__':
    
    filepath = 'C:\\data\\ucddb\\drowsy_awake_7feat.mat'
    feats = loadData(filepath)    
    
#    # TEST: Add new features...
#    feats['EEG_alpha/beta'] = feats.EEG_alpha/feats.EEG_beta
#    feats['EEG_theta/beta'] = feats.EEG_theta/feats.EEG_beta
#    feats['EEG_gamma/beta'] = feats.EEG_gamma/feats.EEG_beta
#    feats['EEG_gamma/delta'] = feats.EEG_gamma/feats.EEG_delta    
    
    
    # 0. Descriptive analysis of data #########################################
    # Boxplots of features against class
    feats.groupby('label').boxplot()
    #plt.xticks(rotation=90)
    #feats.boxplot(by='label')
    
    # Distributions (histograms)
    feats.groupby('label').hist(alpha=0.4)
    
    # Correlation with the labels
    plt.figure()
    feats.corr()['label'].plot(kind='bar')
    plt.title('Pearson correlation of the features with the labels')
    plt.ylim(ymax = 1.1, ymin = -0.5)
    
    # Scatter plot of features
    colors = np.where(feats.label > 0.5, 'r', 'g')
    feats.plot(kind='scatter', x='EEG_fractal_exponent', y='EEG_gamma', alpha=0.3, c=colors, s=50)
    feats.plot(kind='scatter', x='EEG_delta', y='EEG_fractal_exponent', alpha=0.3, c=colors, s=50)

    # Plot principal components
    pca = decomposition.PCA(n_components=3)
    pca.fit(feats.iloc[:,:-1])
    PCs = pca.transform(feats.iloc[:,:-1])
    PCs = pd.DataFrame(PCs, columns=['PC1','PC2','PC3'])
    PCs['label'] = feats['label']
    PCs.plot(kind='scatter', x='PC1', y='PC2', alpha=0.3, c=colors, s=50)
    
    # ANOVA to see if the features are relevant
    # TODO!!
    
    
    # 0.5 Prepare classification    
    rng = 42 # Set random number
    # Separate data in folds
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        feats.iloc[:,:-1], feats.label, test_size=0.15, random_state=rng)
    
    # 1. LDA ##################################################################    
    
    clf = LDA()
    
    # Since no hyperparameter search is necessary with LDA, no need to cross-validate
    #trainScoresLDA = cross_validation.cross_val_score(clf, X_train, y_train, cv=5, verbose=5)    
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_score(y_test, y_pred)
    #print(classification_report(y_test, y_pred))    
    
    # 2. Logistic regression ##################################################
    
    clf = linear_model.LogisticRegression(penalty='l2', C=1.0)
    scoresLR = cross_validation.cross_val_score(clf, X_train, y_train, cv=5, verbose=5)    

    resultsLR = pd.Series(scoresLR)
    
    # 3. SVM  #################################################################
#    
#    # Set up the hyperparameter search
#    svc = svm.SVC()
##    svc.fit(feats.iloc[:,:-1], feats.label)
#    
#    param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},]
#                 # {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
#    clf = GridSearchCV(estimator=svc, param_grid=param_grid, cv=3)#, n_jobs=-1)
##    
##    clf.fit(feats.iloc[:,:-1], feats.label)
##    
##    target_names = ['class 0', 'class 1', 'class 2']
##    print(classification_report(y_true, y_pred, target_names=target_names))
##    
##    # Cross-validate
#    svmResults = cross_validation.cross_val_score(clf, feats.iloc[:,:-1], feats.label, cv=5, verbose=5)


    # 4. GMM ##################################################################
#    g = mixture.GMM(n_components=2)
#    g.fit(feats.iloc[:,:-1])

    # 5. Random forest ########################################################

    # 6. Neural networks ######################################################
    # a) Vanilla NN (~1-3 layers)
    # b) Autoencoder pre-training?
    # c) Convolutions?

    # 7. K-nearest neighbors ##################################################

    # 8. Naive Bayes ##########################################################

    # 9. Ensemble (bagging, adaboost) #########################################