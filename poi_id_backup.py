#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot


sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# added by me.. (py file shall be run on its own)
sys.path.append("myedits/")
from pkl_edit import pkledit
# end.

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','from_poi_%','to_poi_%'] # You will need to use more features
'''
features_list = ['poi', 'mean_top2000_80%tfidfsums', 'salary', 'total_payments', \
'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', \
'other', 'long_term_incentive', 'restricted_stock', 'to_messages', 'from_poi_to_this_person', \
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
'''

### Load the dictionary containing the dataset
#with open(pkledit("final_project_dataset"), "rb") as data_file:
#    data_dict = pickle.load(data_file)
data_dict = pickle.load(open('UPDATED_data_dict.pkl', 'rb'))


### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
[data_dict.pop(key, 0) for key in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']]

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# getting features ready for 2d visualization

#pca = PCA(n_components=2)
#features = pca.fit_transform(features)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# visualizing 2D data

for poi, feats in zip(labels, features):
    frmpoi = feats[0]
    topoi = feats[1]

    plot_color = 'red' if poi == 1 else 'blue'
    matplotlib.pyplot.scatter(frmpoi, topoi, color=plot_color)

matplotlib.pyplot.xlabel("from_poi_%")
matplotlib.pyplot.ylabel("to_poi_%")
matplotlib.pyplot.show()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
'''
#from sklearn.naive_bayes import GaussianN
#clf = GaussianNB()
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


pca = PCA()
scaler = MinMaxScaler()
RFclf = RandomForestClassifier()


estimators = [('pca', pca), ('scaler', scaler), ('RFclf', RFclf)]
pipe = Pipeline(estimators)
grid_param = {
                'pca__n_components': [4, 14],
                'RFclf__min_samples_split': [2, 10, 30],
                'RFclf__bootstrap': ('True', 'False'),
                'RFclf__n_estimators':[150, 200, 250]
        }

clf = GridSearchCV(pipe, grid_param, n_jobs=-1)
'''
#print(pipe.get_params)



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
'''
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)

print('accuracy:', acc)

print(clf.best_params_)
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
