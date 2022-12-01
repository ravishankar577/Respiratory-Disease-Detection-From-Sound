#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# This codebase was to created to make it easier for machine learning researchers to create innovation in the medical field. Combining machine learning with the medical field would decrease false diagnoses and save lives.
# 
# This codebase is focused on **predicting lung diseases** from a breathing sounds dataset. 8 experiments were conducted on 5 different machine learning models. In addition, a novel data augmentation algorithm was performed and tested.
# 
# The following experiments were conducted on 4 classical machine learning methods (decision tree, random forest, SVM, XGBoost):
# 
# 
# *   Using all features to train the models
# *   Using less complex models to decrease overfitting
# * Using class weights to counter dataset unbalancedness
# * Using fewer features to decrease noise in the data
# 
# Experiments on the deep learning model (CNN) were as follows:
# 
# * Using all features to train the model
# * Using class weights to counter dataset unbalancedness
# * Using a novel data augmentation algorithm
# 
# 

# # Project licence

# *MIT License*
# 
# *Copyright (c) 2020 Richard Annilo*
# 
# *Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:*
# 
# *The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.*
# 
# *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.*

# # Imports

# In[1]:


#Data analysis
import pandas as pd
import numpy as np

#Graphing
import matplotlib.pyplot as plt
import seaborn as sns

#Standardizing
from sklearn import preprocessing

#Principal component analysis
from sklearn.decomposition import PCA

#Machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

#Cross validation
from sklearn.model_selection import cross_validate, cross_val_predict

#For evaluating the model
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

from random import randint, shuffle, seed


# # Classical machine learning (ML)

# ## Preprocessing

# In[2]:


#Reading data
all_data = pd.read_csv("dataframes/all_tabular_data.csv")
classes = pd.read_pickle("dataframes/class-codes-for-per-patient-mfcc.pkl")


# In[3]:


#Here are all tabular features and diagnoses for all patients
#This is pre-prepared
all_data


# In[4]:


#Here are the features included
print(list(all_data.columns))
len(list(all_data.columns))


# Let's look at the distribution of classes.

# In[5]:


class_list = list(classes[0]) #For convenience
print(class_list)
classes


# In[6]:


indexes = list(all_data.diagnosis.value_counts().index)
value_counts = list(all_data.diagnosis.value_counts())
x_vals = [classes.loc[i][0] for i in indexes]
plt.figure(figsize=(16,5))
plt.bar(x_vals, value_counts)


# Does not look too good. COPD has more than 10 times more patients in our dataset than Bronchiolitis or Pneumonia. This will most likely impact the performance of our models, because they will want to focus on predicting COPD while ignoring Bronchiolitis.

# In[7]:


all_data.diagnosis.value_counts()


# The results are gathered by running 5-fold cross validation 10 times and box-plotting the results. This is done in order for the results to be as robust as possible and not be varying because of randomness. Only one 5-fold CV appeared to be quite susceptible to random fluctuations.
# 
# The main metric to compare models against each other was chosen to be the macro average of F1-score. F1-score because it takes into account both recall and precision. Macro average, because it takes each class with equal weight no matter how many samples there are in each class. Because our data is very imbalanced.

# In[54]:


def normalize_and_shuffle(all_data):
    _all_data = all_data.sample(frac=1)
    data_x = _all_data.iloc[:, 1:-1]
    data_y = _all_data.diagnosis
    standard_data_x = preprocessing.scale(data_x.to_numpy())
    return data_x, standard_data_x, data_y

data_x, standard_data_x, data_y = normalize_and_shuffle(all_data)


# ## Experiment 1. Training and evaluating with all of the data

# In[9]:


models = [
    ("Decision tree", DecisionTreeClassifier()),
    ("Random forest", RandomForestClassifier(n_estimators=300, max_samples=0.3, max_features=0.9)),
    ("XGBoost", XGBClassifier(learning_rate=0.1, n_estimators=91, min_child_weight=3, max_depth=4, gamma=0.1, reg_alpha=0.01)),
    ("Random forest", RandomForestClassifier(n_estimators=300)),
    ("SVM", SVC(kernel="sigmoid", gamma=0.02, C=46))
]


# In[10]:


scoring = {'f1_macro': 'f1_macro',
          'f1_micro': 'f1_micro',
          'accuracy':'accuracy'}


# In[11]:


#The following are global variables

times = 10 #The number of times we will run cross validation

f1_scores = {} #F1-macro-averages
f1_micros = {} #F1-micro-averages
f1_train_scores = {} #F1-train-marco-averages
y_preds = {} #Predicted diagnoses
estimators = {} #The models that are trained during cross validation
y_actuals = list(data_y)*times #The acutual diagnoses


# In[12]:


for item in models:    
    name = item[0]
    model = item[1]
    temp_f1 = []
    temp_f1_micro = []
    temp_y_preds = []
    temp_f1_train = []
    temp_estimators = []
    
    for _ in range(times):
        #Random shuffle
        data_x, standard_data_x, data_y = normalize_and_shuffle(all_data)

        #standard_data_x, data_y = normalize_and_shuffle(all_data)
        
        cv = cross_validate(estimator=model, 
                            X=standard_data_x, 
                            y=data_y, 
                            cv=5, #Number of folds
                            scoring=scoring, 
                            verbose=0, 
                            return_train_score=True,
                            return_estimator=True) #Returns scores
        cv_pred = cross_val_predict(estimator=model, X=standard_data_x, y=data_y, cv=5) #Returns predictions
        temp_f1.extend(cv["test_f1_macro"])
        temp_f1_micro.extend(cv["test_f1_micro"])
        temp_f1_train.extend(cv["train_f1_macro"])
        temp_estimators.extend(cv["estimator"])
        temp_y_preds.extend(cv_pred)
        
    f1_scores[name] = temp_f1
    f1_micros[name] = temp_f1_micro
    f1_train_scores[name] = temp_f1_train
    y_preds[name] = temp_y_preds
    if (item == "SVM"):
        print(temp_y_preds)
    estimators[name] = temp_estimators


# In[13]:


#Plots boxplots
def plot_boxplots(xticks, data, font_scale = 1.5, figsize = (16,7), title = "Untitled graph", ylabel = "F1 macro-average", rotation = 0, baseline = 0.1, models=None, order=None, palette=None):
    
    temp_xticks = xticks.copy()
    temp_data = data.copy()
    
    if (models != None):
        for x in xticks:
            if (x not in models):
                i = xticks.index(x)
                temp_xticks.remove(x)
                temp_data.remove(data[i])
    
    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)
    
    all_pal = {"Decision tree": "maroon", 
              "Decision tree (class weights)": "indianred", 
              "Decision tree (fewer features)": "r", 
              "SVM": "darksalmon", 
              "SVM (class weights)": "coral", 
              "SVM (fewer features)": "orangered", 
              "XGBoost": "royalblue", 
              "XGBoost (class weights)": "steelblue", 
              "XGBoost (fewer features)": "skyblue", 
              "Random forest": "darkgreen", 
              "Random forest (class weights)": "forestgreen", 
              "Random forest (fewer features)": "lightgreen"}

    graph = sns.violinplot(data=temp_data, order=order, palette=palette, inner="points")
    plt.xticks(plt.xticks()[0], temp_xticks)
    
    graph.axhline(baseline, color="black", linestyle="--")
    graph.text(x = -0.4, y = baseline+0.02, s="majority class baseline", fontsize=15)
    
    plt.title(title)
    plt.ylabel(ylabel)
    graph.set_xticklabels(graph.get_xticklabels(), rotation=rotation)


# In[14]:


#Plots confusion matrices
def plot_confusion_matrix(y_pred, y_true, classes, title = "Untitled confusion matrix", rotation = 0):
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    df_cm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    plt.title(title)
    graph = sns.heatmap(df_cm, annot=True, fmt='g')
    graph.set_xticklabels(graph.get_xticklabels(), rotation=rotation)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")


# In[15]:


#Sorting dictionary by average value and separating keys and values.
def prepare_dict(d):
    #Source: https://stackoverflow.com/questions/30558440/sorting-a-dictionary-by-average-of-list-in-value
    keys = []
    for key in sorted(d, key=lambda k: sum(d[k]) / len(d[k]), reverse=True):
        keys.append(key)
    values = [d[key] for key in keys]
    return keys, values


# #### Plotting results

# In[16]:


keys, values = prepare_dict(f1_scores)


# In[17]:


for key, value in zip(keys, values):
    print(key, round(np.array(value).mean(), 4), "+/-", round(np.array(value).std(), 4))


# In[18]:


plot_boxplots(keys, values, title="Test performances using all features", rotation=90)


# In[19]:


keys, values = prepare_dict(f1_train_scores)
plot_boxplots(keys, values, title="Training performances using all features", ylabel="F1 macro-averages", baseline = 0.1)


# In[20]:


keys, values = prepare_dict(f1_micros)
plot_boxplots(keys, values, title="Test micro-average scores using all features", ylabel="F1 micro-averages", baseline = 0.5)


# #### Plotting confusion matrices

# In[21]:


plot_confusion_matrix(y_preds["SVM"], y_actuals, class_list, rotation = 20, title="SVM confusion matrix with all features")


# In[22]:


plot_confusion_matrix(y_preds["Decision tree"],
                      y_actuals,
                      class_list,
                      rotation = 20,
                      title="Decision tree with maximum features")


# In[23]:


plot_confusion_matrix(y_preds["Random forest"],
                      y_actuals,
                      class_list,
                      rotation = 20,
                      title="Random forest using all features")


# In[24]:


plot_confusion_matrix(y_preds["XGBoost"],
                      y_actuals,
                      class_list,
                      rotation = 20,
                      title="XGBoost with maximum features")


# #### PCA analysis

# In[25]:


def get_pca_scatterplot(principal_components, data_y, class_list, title="Untitled scatterplot"):
    scat_data = principal_components.copy()
    scat_data["diagnosis"] = data_y
    scat_data.set_index("diagnosis", inplace=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(6):
        sns.scatterplot(scat_data.loc[i, :][0], scat_data.loc[i, :][1], s=70)
    sns.set(font_scale=1.3)
    ax.set(ylabel = "", xlabel= "")
    legend1 = ax.legend(class_list,
                        loc="lower right", title="Classes")
    plt.title(title)
    plt.show()


# In[26]:


#temp_data = data_x[["crackles", "wheezes", "age", "root_mean_square_mean"]]
temp_data = standard_data_x


# In[27]:


n_components = 8
pca = PCA(n_components)
principal_components = pca.fit_transform(temp_data)
print("The", n_components, "components account for", np.sum(pca.explained_variance_ratio_), "of variance")
principal_components = pd.DataFrame(principal_components)
principal_components.head()


# In[28]:


get_pca_scatterplot(principal_components, data_y, class_list, "Principal component scatterplot for all features")


# #### Feature importance analysis

# In[29]:


#Sum of all importances for all estimators used in cross validation
importances = [0] * len(classes)
for est in estimators["Decision tree"]:
    importance = est.feature_importances_
    for i in range(len(importances)):
        importances[i] += importance[i]


# In[30]:


sorted_indices = np.argsort(importances)
sorted_importances = np.sort(importances)


# In[33]:


plt.figure(figsize=(12,12))

plt.title('Feature Importances for Decision tree')
plt.barh(range(len(sorted_indices)), sorted_importances, color='b', align='center')
plt.yticks(range(len(sorted_indices)), [class_list[i] for i in sorted_indices])
plt.xlabel('Relative Importance')
plt.show()


# In[34]:


#Sum of all importances for all estimators used in cross validation
importances = [0] * len(classes)
for est in estimators["Random forest"]:
    importance = est.feature_importances_
    for i in range(len(importances)):
        importances[i] += importance[i]


# In[35]:


sorted_indices = np.argsort(importances)
sorted_importances = np.sort(importances)


# In[36]:


plt.figure(figsize=(12,12))

plt.title('Feature Importances for Random forest')
plt.barh(range(len(sorted_indices)), sorted_importances, color='b', align='center')
plt.yticks(range(len(sorted_indices)), [class_list[i] for i in sorted_indices])
plt.xlabel('Relative Importance')
plt.show()


# From this we learn the following:
# 
# 1. RF and XGBoost are performing worse than DT, because they focus too much on the **majority classes**. A solution to this would be to introduce **class weights**.
# 2. There are a lot of **irrelevant features**. Removing them might improve the results.
# 
# The following 2 experiments aim to amend these issues.

# ## Experiment 2. Introducing class weights to models

# In[37]:


#Generating class weights for each data instance
#Code source "https://stackoverflow.com/questions/45811201/how-to-set-weights-in-multi-class-classification-in-xgboost-for-imbalanced-data"
def CreateBalancedSampleWeights(y_train, largest_class_weight_coef):
    classes = np.unique(y_train, axis = 0)
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key : value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]] * 9999999 #largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]
    return sample_weights

largest_class_weight_coef = data_y.value_counts(normalize=1).iloc[0] #occurance rate of most frequent class
weights = CreateBalancedSampleWeights(data_y, largest_class_weight_coef)


# In[38]:


models = [
    ("Decision tree (class weights)", DecisionTreeClassifier(class_weight="balanced")),
    ("Random forest (class weights)", RandomForestClassifier(n_estimators=250, class_weight="balanced", max_samples=0.3, max_features=0.9)),
    ("XGBoost (class weights)", XGBClassifier(weigths=weights, learning_rate=0.1, n_estimators=91, min_child_weight=3, max_depth=4, gamma=0.1, reg_alpha=0.01)),
    ("SVM (class weights)", SVC(kernel="sigmoid", C=12, gamma=0.0055, class_weight="balanced")),
]


# In[39]:


for item in models:    
    name = item[0]
    model = item[1]
    temp_f1 = []
    temp_f1_micro = []
    temp_y_preds = []
    temp_f1_train = []
    temp_estimators = []
    
    for _ in range(times):
        standard_data_x, data_y = normalize_and_shuffle(all_data)
        cv = cross_validate(estimator=model, 
                            X=standard_data_x, 
                            y=data_y, 
                            cv=5, 
                            scoring=scoring, 
                            verbose=0, 
                            return_train_score=True,
                            return_estimator=True)
        cv_pred = cross_val_predict(estimator=model, X=standard_data_x, y=data_y, cv=5)
        temp_f1.extend(cv["test_f1_macro"])
        temp_f1_micro.extend(cv["test_f1_micro"])
        temp_f1_train.extend(cv["train_f1_macro"])
        temp_estimators.extend(cv["estimator"])
        temp_y_preds.extend(cv_pred)
        
    f1_scores[name] = temp_f1
    f1_micros[name] = temp_f1_micro
    f1_train_scores[name] = temp_f1_train
    y_preds[name] = temp_y_preds
    estimators[name] = temp_estimators


# In[40]:


keys, values = prepare_dict(f1_scores)


# In[41]:


plot_boxplots(keys, values, title="Model performances on all data", rotation=10, models=["Decision tree (class weights)", "Random forest (class weights)", "XGBoost (class weights)", "SVM (class weights)"])


# In[42]:


plot_boxplots(keys, values, title="Model performances on all data", models=["Decision tree (class weights)", "Decision tree"], figsize=(8,5))


# In[43]:


plot_boxplots(keys, values, title="Model performances on all data", models=["Random forest (class weights)", "Random forest (size limit)", "Random forest"])


# In[44]:


#XGBOOST BOXPLOTS
plot_boxplots(keys, values, title="Model performances on all data", models=["XGBoost (class weights)", "XGBoost"])


# In[45]:


plot_boxplots(keys, values, title="Model performances on all data", models=["SVM (class weights)", "SVM"])


# **Results**: Decision tree, Random forest and SVM improved from introducing class weights. 

# ## Experiment 3. Training with fewer features

# In[47]:


#Sum of all importances for all estimators used in cross validation
importances = [0] * len(classes)
for est in estimators["Decision tree (class weights)"]:
    importance = est.feature_importances_
    for i in range(len(importances)):
        importances[i] += importance[i]
        
sorted_indices = np.argsort(importances)
sorted_importances = np.sort(importances)

plt.figure(figsize=(12,12))

plt.title('Feature Importances for Decision tree (class weights)')
plt.barh(range(len(sorted_indices)), sorted_importances, color='b', align='center')
plt.yticks(range(len(sorted_indices)), [class_list[i] for i in sorted_indices])
plt.xlabel('Relative Importance')
plt.show()


# The following features have been selected because they show the largest difference between classes on the PCA plot.

# In[48]:


fewer_features = ["wheezes","crackles","age", "root_mean_square_mean", "spectral_entropy", "spectral_flatness_mean", "zero_crossing_rate"]


# In[55]:


sml_data_x = data_x[fewer_features]
sml_data_x.head()


# In[59]:


pca = PCA(2)
principal_components = pca.fit_transform(sml_data_x)
principal_components = pd.DataFrame(principal_components)
print("PCA accounts for", sum(pca.explained_variance_ratio_), "of the variance")


# In[60]:


get_pca_scatterplot(principal_components, data_y, class_list, "Principal component scatterplot for fewer features")


# Do these class similarities make sense?
# 1. Healthy people are similar to URTI. That makes sense. There is practically no way you can tell apart a person with URTI and a healthy person by listening to their lungs.
# 
# 2. COPD patients are similar to Pneumonia. This makes some sense. Both diseases can have crackles and wheezes.
# 
# 3. COPD patients are similar to Bronchiectasis. This makes some sense, since Bronchiectasis can be a result of COPD.

# In[61]:


#Standardizing data, necessary for SVM
standard_sml_data_x = preprocessing.scale(sml_data_x.to_numpy())
standard_sml_data_x


# In[62]:


models = [
    ("Decision tree (fewer features)", DecisionTreeClassifier()),
    ("Random forest (fewer features)", RandomForestClassifier(n_estimators=300, max_samples=0.3, max_features=0.8)),
    ("XGBoost (fewer features)", XGBClassifier(learning_rate=0.1, n_estimators=62, min_child_weight=3, max_depth=3, gamma=0)),
    ("SVM (fewer features)", SVC(kernel="sigmoid", C=40, gamma=0.025))
]


# In[63]:


sml_data_x["diagnosis"] = data_y


# In[64]:


for item in models:    
    name = item[0]
    model = item[1]
    temp_f1 = []
    temp_f1_micro = []
    temp_y_preds = []
    temp_f1_train = []
    temp_estimators = []
    
    for _ in range(times):
        cv = cross_validate(estimator=model, 
                            X=standard_sml_data_x, 
                            y=data_y, 
                            cv=5, 
                            scoring=scoring, 
                            verbose=0, 
                            return_train_score=True,
                            return_estimator=True)
        cv_pred = cross_val_predict(estimator=model, X=sml_data_x, y=data_y, cv=5)
        temp_f1.extend(cv["test_f1_macro"])
        temp_f1_micro.extend(cv["test_f1_micro"])
        temp_f1_train.extend(cv["train_f1_macro"])
        temp_estimators.extend(cv["estimator"])
        temp_y_preds.extend(cv_pred)
        
    f1_scores[name] = temp_f1
    f1_micros[name] = temp_f1_micro
    f1_train_scores[name] = temp_f1_train
    y_preds[name] = temp_y_preds
    estimators[name] = temp_estimators


# In[65]:


keys, values = prepare_dict(f1_scores)


# In[66]:


plot_boxplots(keys, values, rotation = 20, title = "Model performances", models=["Random forest (fewer features)", "XGBoost (fewer features)", "SVM (fewer features)", "Decision tree (fewer features)"])


# In[67]:


plot_boxplots(keys, values, rotation = 20, title = "Model performances", models=["Random forest", "Random forest (class weights)", "Random forest (fewer features)", "Random forest (size limit)"])


# In[68]:


plot_boxplots(keys, values, rotation = 20, title = "Model performances", models=["SVM", "SVM (class weights)", "SVM (fewer features)"])


# **Results**: Having fewer features helps all of the models.

# ### Conclusion

# In[69]:


keys, values = prepare_dict(f1_scores)
for key, values in zip(keys, values):
    print(key, round(np.array(values).mean(), 4), "+/-", round(np.array(values).std(), 4))


# In[70]:


plot_boxplots(keys, values, rotation = 90, title = "Model performances")


# The best versions of each model are the following.

# In[71]:


print("Decision tree (class weights) score is", round(np.average(f1_scores["Decision tree (class weights)"]), 4))
print("SVM (fewer features)", round(np.average(f1_scores["SVM (fewer features)"]), 4))
print("XGBoost (fewer features) score is", round(np.average(f1_scores["XGBoost (fewer features)"]), 4))
print("Random forest (fewer features) score is", round(np.average(f1_scores["Random forest (fewer features)"]), 4))


# Let's see if deep learning can do any better

# # Deep learning

# In[73]:


#Random operations
from random import randint, shuffle

from sklearn.metrics import f1_score

#Modules for the CNN
from keras.utils import to_categorical #Data prepration
from keras.models import Sequential #Layers
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam #Optimizer
from keras import regularizers #Regularizer
from keras.callbacks import EarlyStopping

#Item-wise addition of two lists
from operator import add


# ### Loading data

# In[74]:


#Patients split into 5 folds. Selected such that each fold contains patient with every different disease.
patient_folds = np.load("dataframes/equal_patient_folds.npy", allow_pickle=True)
#Pre-made dataframe containing all patients with all MFCCs
patient_mfccs_and_diagnosis = pd.read_pickle("dataframes/per-patient-mfccs")
#Class names
classes = pd.read_pickle("dataframes/class-codes-for-per-patient-mfcc.pkl")


# In[75]:


patient_mfccs_and_diagnosis


# In[76]:


classes = list(classes[0])
classes


# In[77]:


patient_folds


# In[78]:


#Contains 20 MFCCs (one for each soundfile) with the size 40x431.
temp_all = []
for _, row in patient_mfccs_and_diagnosis.iterrows():
    temp_all.extend(row.mfcc)
np.array(temp_all).shape


# In[79]:


temp_diagnosis_per_soundfile = []
for _, row in patient_mfccs_and_diagnosis.iterrows():
    temp_diagnosis_per_soundfile.extend(len(row.mfcc)*[classes[row.diagnosis]])


# In[80]:


plt.figure(figsize=(10,4))
pd.array(temp_diagnosis_per_soundfile).value_counts().plot(kind="bar", title="Diagnosis counts for soundfiles")


# ### Training

# In[81]:


#Storing hyperparameters here
class Config(object):
    def __init__(self, n_mfcc, max_frames, sample_rate, max_audio_duration, batch_size, epochs):
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.max_audio_duration = max_audio_duration
        self.max_audio_length = max_audio_duration * sample_rate
        self.batch_size = batch_size
        self.epochs = epochs


# In[82]:


config = Config(n_mfcc=40, 
                max_frames=862, 
                sample_rate = 11025, 
                max_audio_duration = 20, 
                batch_size = 64, 
                epochs = 700)


# In[90]:


#Returns CNN model

def get_model(config):
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(40,431,1), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=64, kernel_size=2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.4))
    
    model.add(GlobalAveragePooling2D())

    model.add(Dense(len(classes), activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  metrics=['accuracy'], 
                  optimizer=Adam(amsgrad=True)) 
    
    return model


# In[91]:


#Global variables
cnn_f1_scores = {}
cnn_f1_train_scores = {}
cnn_y_preds = {}
cnn_y_train_preds = {}
cnn_histories = {}


# In[92]:


#These are the actual diagnoses
temp_df = patient_mfccs_and_diagnosis.copy()
temp_df.set_index("patient", inplace=True) #To be able to get row by patient
cnn_y_actuals = []
for fold in patient_folds:
    for patient in fold:
        cnn_y_actuals.append(temp_df.loc[patient].diagnosis)


# In[93]:


patient_folds


# In[94]:


data_y


# In[95]:


#Training + testing
#5 foldy CV

#Predicting the diagnosis of each patient 
#by summing the prediction probabilities 
#of each soundfile for each patient

def get_results_with_model(dataframe, patient_folds, class_weights = None, fold_nr=None, verbose=0):
    histories = []
    predicted_y = []
    train_predicted_y = []
    f1_scores = []
    f1_train_scores = []
    
    if fold_nr is not None:
        patient_folds = [patient_folds[fold_nr]]
    for _ in range(9):
        for test_patients in patient_folds:
            X_train = []
            y_train = []
            test_rows = []
            train_rows = []

            #Gathering train and test rows
            for _, row in dataframe.iterrows():
                if (row.patient in test_patients):
                    test_rows.append(row)
                else:
                    train_rows.append(row)

            #Data preparation
            shuffle(train_rows)

            for row in train_rows:
                np.array(row.mfcc)
                X_train.extend(row.mfcc)
                y_train.extend([row.diagnosis]*len(row.mfcc))

            X_train = np.array(X_train)
            X_train = np.expand_dims(X_train, -1)

            y_train = np.array(y_train)
            y_train = to_categorical(y_train)

            val_split = int(len(X_train)*0.2)
            X_val = X_train[:val_split]
            y_val = y_train[:val_split]
            X_train = X_train[val_split:]
            y_train = y_train[val_split:]

            #Training
            model = get_model(config)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=50)
            history = model.fit(x=X_train, 
                                y=y_train, 
                                batch_size=config.batch_size, 
                                epochs=config.epochs, 
                                validation_data=(X_val, y_val), 
                                verbose=verbose,
                                callbacks=[es],
                               class_weight=class_weights)    
            histories.append(history)

            #Getting results with the model
            cur_fold_test_predicted = []
            cur_fold_test_actuals = []
            cur_fold_train_predicted = []
            cur_fold_train_actuals = []

            #Test results
            for row in test_rows:
                mfccs = row.mfcc
                patient = row.patient
                summed_predictions = [0]*6 #Sum of predictions for one patient
                for mfcc in mfccs: #All soundfiles for one patient
                    x = np.array([mfcc])
                    x = np.expand_dims(x, -1)
                    prediction = model.predict(x)
                    summed_predictions = list(map(add, summed_predictions, prediction))
                cur_fold_test_predicted.append(np.argmax(summed_predictions)) #Diagnosis with highest prediction probability
                cur_fold_test_actuals.append(row.diagnosis)

            #Train results
            for row in train_rows:
                mfccs = row.mfcc
                patient = row.patient
                summed_predictions = [0]*6
                for mfcc in mfccs:
                    x = np.array([mfcc])
                    x = np.expand_dims(x, -1)
                    prediction = model.predict(x)
                    summed_predictions = list(map(add, summed_predictions, prediction))
                cur_fold_train_predicted.append(np.argmax(summed_predictions)) 
                cur_fold_train_actuals.append(row.diagnosis)

            f1_scores.append(f1_score(y_pred=cur_fold_test_predicted, y_true=cur_fold_test_actuals, average="macro"))
            f1_train_scores.append(f1_score(y_pred=cur_fold_train_predicted, y_true=cur_fold_train_actuals, average="macro"))
            predicted_y.append(cur_fold_test_predicted) 
            train_predicted_y.append(cur_fold_train_predicted)
    return histories, predicted_y, train_predicted_y, f1_scores, f1_train_scores


# In[96]:


#Remove to train on all folds. Currently training on one fold for speed.
histories, predicted_y, train_predicted_y, f1_scores, f1_train_scores = get_results_with_model(patient_mfccs_and_diagnosis, patient_folds, verbose=0, fold_nr=1)


# In[97]:


cnn_f1_scores["CNN"] = f1_scores
cnn_f1_train_scores["CNN"] = f1_train_scores
cnn_y_preds["CNN"] = predicted_y
cnn_y_train_preds["CNN"] = train_predicted_y
cnn_histories["CNN"] = histories


# In[99]:


np.mean((np.array(cnn_f1_scores["CNN"])))


# In[100]:


for i, history in enumerate(cnn_histories["CNN"]):
    #summarizing history of training
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title("Training and validation accuracy for CNN. Fold number " + str(i))
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


# In[101]:


keys, values = prepare_dict(cnn_f1_scores)
plot_boxplots(keys, values, title="CNN performaces", figsize=(7, 6), models=["CNN"])


# In[102]:


for key, values in zip(keys, values):
    print(key, round(np.array(values).mean(), 4), 
          "+/-", round(np.array(values).std(), 4))


# ### Training with class weights

# In[104]:


class_weights = {0:(793/16), 1:(793/13), 2:1, 3:(793/35), 4:(793/37), 5:(793/23)}


# In[ ]:


histories, predicted_y, train_predicted_y, f1_scores, f1_train_scores = get_results_with_model(patient_mfccs_and_diagnosis, patient_folds, class_weights = class_weights, verbose=0)

cnn_f1_scores["CNN (class weights)"] = f1_scores
cnn_f1_train_scores["CNN (class weights)"] = f1_train_scores
cnn_y_preds["CNN (class weights)"] = predicted_y
cnn_y_train_preds["CNN (class weights)"] = train_predicted_y
cnn_histories["CNN (class weights)"] = histories


# In[ ]:


cnn_f1_scores


# In[ ]:


keys, values = prepare_dict(cnn_f1_scores)
plot_boxplots(keys, values, title="CNN performances", figsize=(9, 7), models=["CNN", "CNN (class weights)"])
# Miks tulemused erinevad: vähe andmepunkte - eri foldid on erinevad, erinevad maailmad.  
# Miks tulemused erinevad: lisaks, närvivõrkude eri alguspunktid on erinevad. 


# In[ ]:


plot_confusion_matrix(y_pred = [item for sublist in cnn_y_preds["CNN (class weights)"] for item in sublist], 
                      y_true=cnn_y_actuals, classes=list(classes[0]), title="CNN (class weights) confusion matrix", rotation=30)


# In[ ]:


for i, history in enumerate(cnn_histories["CNN (class weights)"]):
    #summarizing history of training
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title("Accuracy for CNN (class weights). Fold number " + str(i))
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


# ### Training with augmented data

# In[ ]:


augmented_data = pd.read_pickle(root + "dataframes/augmented-per-patient-mfccs")
augmented_data


# In[ ]:


cnn_f1_scores["CNN augmented data"] = []
cnn_f1_train_scores["CNN augmented data"] = []
cnn_y_preds["CNN augmented data"] = []
cnn_y_train_preds["CNN augmented data"] = []
cnn_histories["CNN augmented data"] = []


# In[ ]:


# Currently training on one fold only. Remove "fold_nr" variale to run on all folds (might take a while)
histories, predicted_y, train_predicted_y, f1_scores, f1_train_scores = get_results_with_model(augmented_data, patient_folds, fold_nr = 2)

cnn_f1_scores["CNN augmented data"].extend(f1_scores)
cnn_f1_train_scores["CNN augmented data"].extend(f1_train_scores)
cnn_y_preds["CNN augmented data"].extend(predicted_y)
cnn_y_train_preds["CNN augmented data"].extend(train_predicted_y)
cnn_histories["CNN augmented data"].extend(histories)


# In[ ]:


cnn_f1_train_scores


# In[ ]:


for i, history in enumerate(cnn_histories["CNN augmented data"]):
    #summarizing history of training
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title("Accuracy for CNN (augmented data). Fold number " + str(i))
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


# In[ ]:


cnn_f1_scores


# In[ ]:


plot_confusion_matrix(y_pred = [2, 2, 0, 3, 2, 3, 3, 2, 2, 4, 4, 0, 3, 1, 3, 3, 3, 1, 3, 2, 2, 2, 3, 2, 1, 4, 2, 0, 3, 1, 1, 2, 2, 2, 5, 4, 4, 2, 2, 1, 4, 3, 2, 3, 3, 1, 4, 4, 2, 2, 5, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 2, 4, 2, 2, 2, 0, 4, 2, 4, 2, 4, 5, 0, 0, 2, 2, 2, 2, 3, 2, 5, 5, 2, 2, 2, 2, 2, 2, 2, 3, 0, 2, 2, 3, 2, 3, 2, 2, 5, 5, 5, 2, 4, 4, 2, 2, 2, 2, 0, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 3, 4, 4]
, 
                      y_true=cnn_y_actuals, classes=list(classes[0]), title="Confusion matrix of CNN (data augmentation)", rotation=30)


# In[ ]:


keys, values = prepare_dict(cnn_f1_scores)
plot_boxplots(keys, values, title="All CNN models test results")


# # Results + Project poster

# Here is an high-level overview of the project and all of the results for all of the experiments

# ![alt text](https://i.imgur.com/SHuBk5C.jpg)
