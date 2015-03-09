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

from sklearn import decomposition, mixture, lda, svm, linear_model, neighbors, naive_bayes, ensemble
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib

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
    
def descriptiveAnalysis(feats):
    """Descriptive analysis of data"""
    
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
    
def gridSearch_thenTest(clf, param_grid, cv, X_train, y_train, X_test, y_test):
    """First perform a grid search of hyperparameters in param_grid, then test
    on the test set. 
    
    Taken from http://scikit-learn.org/stable/auto_examples/grid_search_digits.html#example-grid-search-digits-py
    """
    
    score = 'accuracy'
    
    print('# Tuning hyper-parameters for %s' %score)
    print()

    clf = GridSearchCV(clf, param_grid, cv=cv, scoring=score, n_jobs=-1, verbose=True)
    clf.fit(X_train, y_train)

    print('Best parameters set found on development set:')
    print()
    print(clf.best_estimator_)
    print()
    print('Grid scores on development set:')
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print('%0.3f (+/-%0.03f) for %r'
              % (mean_score, scores.std() / 2, params))
    print()

    print('Detailed classification report:')
    print()
    print('The model is trained on the full development set.')
    print('The scores are computed on the full test set.')
    print()
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Accuracy: %0.3f'%accuracy_score(y_test, y_pred))
    print()
    
    return clf
    
if __name__ == '__main__':
    
    #filepath = 'C:\\data\\ucddb\\drowsy_awake_7feat.mat' # Laptop
    filepath = '/home/hubert/data/ucddb/preprocessed/drowsy_awake_7feat.mat' # Desktop (Ubuntu)
    feats = loadData(filepath)    
    
#    # TEST: Add new features...
#    feats['EEG_alpha/beta'] = feats.EEG_alpha/feats.EEG_beta
#    feats['EEG_theta/beta'] = feats.EEG_theta/feats.EEG_beta
#    feats['EEG_gamma/beta'] = feats.EEG_gamma/feats.EEG_beta
#    feats['EEG_gamma/delta'] = feats.EEG_gamma/feats.EEG_delta    
    
#    # TEST: Remove uncorrelated features...
#    feats.drop(['EEG_alpha','EEG_beta'],inplace=True,axis=1)
    
    # TEST: Normalize the features...
    # They are already pretty much z-score normalized... (as was said in the email)
    labels = feats.label
    feats = (feats - feats.mean())/feats.std()
    feats['label'] = labels
    
    # 0. Descriptive analysis of data #########################################
    #descriptiveAnalysis(feats)
    
    # 0.5 Prepare classification    
    # TODO: Handle unbalanced classes!
    rng = 42 # Set random number
    
    # Separate data in train and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        feats.iloc[:,:-1], feats.label, test_size=0.15, random_state=rng)
    
    # Separate train set in stratified folds for cross-validation
    cv = cross_validation.StratifiedKFold(y_train, n_folds=3)
    
    # 0. Random classification of data ##########################################
    
    print('Classifier 0: Random')
    print()
    
    # see http://scikit-learn.org/stable/auto_examples/plot_permutation_test_for_classification.html#example-plot-permutation-test-for-classification-py
    clf = DummyClassifier(strategy='uniform',random_state=rng)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Accuracy: %0.3f'%accuracy_score(y_test, y_pred))
    print()
    
    # 1. LDA ##################################################################    
    
#    print('Classifier 1: LDA')
#    print()
#    
#    clf = lda.LDA()
#    
#    # Since no hyperparameter search is necessary with LDA, no need to cross-validate
#    #trainScoresLDA = cross_validation.cross_val_score(clf, X_train, y_train, cv=5, verbose=5)    
#    
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#    print(classification_report(y_test, y_pred)) 
#    print('Accuracy: %0.3f'%accuracy_score(y_test, y_pred))   
#    print()

    
    # 2. Logistic regression ##################################################
    
#    print('Classifier 2: Logistic regression')
#    print()    
#    
#    clf = linear_model.LogisticRegression(random_state=rng)
#    
#    param_grid = [{'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2', 'l1']},]
#
#    bestLRclf = gridSearch_thenTest(clf, param_grid, cv, X_train, y_train, X_test, y_test)
#    
#    print()

    
    # 3. SVM  #################################################################
    
#    print('Classifier 3: SVM')
#    print()    
#    
#    # TODO: Try LinearSVM with l2 loss (reduces capacity and increases convergence speed)
#    
#    clf = svm.SVC(random_state=rng, tol=0.1)
#    
#    # Set up the hyperparameter search    
#    param_grid = [{'C': [1, 10], 'kernel': ['linear']},
#                  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]    
#
#    bestSVMclf = gridSearch_thenTest(clf, param_grid, cv, X_train, y_train, X_test, y_test)
#    
#    print()
#
##    # Cross-validate
##    svmResults = cross_validation.cross_val_score(clf, feats.iloc[:,:-1], feats.label, cv=5, verbose=5)


    # 4. GMM ##################################################################
    # TODO: Make this one work!    
    
#    print('Classifier 4: GMM')
#    print()
#    
#    clf = mixture.GMM(n_components=2, random_state=rng)
#    param_grid = [{'covariance_type': ['spherical', 'tied', 'diag', 'full']}]
#    
#    clf = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=True)
#    clf.fit(X_train)
#    
#    print('Best parameters set found on development set:')
#    print()
#    print(clf.best_estimator_)
#    print()
#    print('Grid scores on development set:')
#    print()
#    for params, mean_score, scores in clf.grid_scores_:
#        print('%0.3f (+/-%0.03f) for %r'
#              % (mean_score, scores.std() / 2, params))
#    print()
#
#    print('Detailed classification report:')
#    print()
#    print('The model is trained on the full development set.')
#    print('The scores are computed on the full test set.')
#    print()
#    
#    y_pred = clf.predict(X_test)
#    print(classification_report(y_test, y_pred))
#    print('Accuracy: %0.3f'%accuracy_score(y_test, y_pred))
#    print()
#    
#    print()

    # 5. Random forest ########################################################

#    print('Classifier 5: Random Forest')
#    print()
#
#    clf = ensemble.RandomForestClassifier(random_state=rng)
#    param_grid = [{'n_estimators': [300]}]
#    
#    bestRFclf = gridSearch_thenTest(clf, param_grid, cv, X_train, y_train, X_test, y_test)
#    
#    print()

    # 6. Neural networks ######################################################
    # a) Vanilla NN (~1-3 layers)
    # b) Autoencoder pre-training?
    # c) Convolutions?

    # 7. K-nearest neighbors ##################################################
    # Since the data is not too high-dimensional, this might work well
    # TODO: Still, try with as few features as possible

#    print('Classifier 7: K-Nearest Neighbors')
#    print()    
#
#    clf = neighbors.KNeighborsClassifier()
#    param_grid = [{'n_neighbors': range(26,101,5)}] #, 'weights': ['uniform','distance']}]
#
#    bestkNNclf = gridSearch_thenTest(clf, param_grid, cv, X_train, y_train, X_test, y_test)
#    
#    print()

    # 8. Naive Bayes ##########################################################

#    print('Classifier 8: Naive Bayes')
#    print()    
#    
#    clf = naive_bayes.GaussianNB()
#    
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#    print(classification_report(y_test, y_pred)) 
#    print('Accuracy: %0.3f'%accuracy_score(y_test, y_pred))   
#    print()
    

    # 9. Ensemble (bagging, adaboost) #########################################


    # 10. Gradient Boosting Classifier ########################################
    # This was the best classifier in the Kaggle BCI challenge...


# Once the best model is selected:
#   - See the effect of feature selection
#   - Look at the learning curves for different hyperparameters: sklearn.learning_curve.validation_curve
#   - Re-train on the whole dataset! (or at least on the whole training set)


# Save the best model
# savename = '.pkl'
# joblib.dump(bestClf, savename) 


"""
Best models:
- SVM: rbf, C=1000, gamma=0.001 ->
- kNN: n_neighbors = 61 (for all 7 features) -> 0.740
       n_neighbors = 71 (for 5 features) -> 0.740
- RF:  n_estimators = 300 -> 0.742
       n_estimators = 1000 -> 0.745
"""