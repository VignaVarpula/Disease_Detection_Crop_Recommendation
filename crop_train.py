# #importing warnings library to avoid warnings
# import warnings
# warnings.filterwarnings('ignore',category=FutureWarning)
# import pandas as pd
# from IPython.display import display


# data1 = pd.read_csv(r'C:\Youtube\DiseaseDetection\dataset\crop_data1.csv')
# data1.head()
# import pandas as pd

# data2 = pd.read_csv(r'C:\Youtube\DiseaseDetection\dataset\crop_data2.csv')
# data2.head()
# #sanity checkp
# data1.info()
# data2.info()
# #calculating shape of columns
# data1.shape
# data2.shape
# #finding missing value
# data1.isnull().sum()
# data2.isnull().sum()
# #finding duplicates
# data1.duplicated().sum()
# data2.duplicated().sum()
# data2.drop_duplicates(subset = ['N','P','K','temperature','humidity','ph','rainfall'],keep = 'first',inplace = True)
# data2.duplicated().sum()
# #identifiying Garbage values
# for i in data1.select_dtypes(include = 'object').columns:
#     print(data1[i].value_counts())
#     print("***"*10)
# for i in data2.select_dtypes(include = 'object').columns:
#     print(data2[i].value_counts())
#     print("***"*10)


# #descriptive statistics
# data1.describe()
# #transposing the data
# data1.describe().T
# data2.describe()
# data2.describe().T
# #descriptive statistics of object column
# data1.describe(include = 'object')
# data2.describe(include = 'object')
# #histogram to understand the distribution
# import seaborn as sns
# from matplotlib import pyplot as plt
# for i in data1.select_dtypes(include = ['number']).columns:
#     sns.histplot(data = data1,x=i)
#     plt.show()
# for i in data2.select_dtypes(include = 'number').columns:
#     sns.histplot(data = data2,x=i)
#     plt.show()

# from scipy import stats
# import numpy as np
# z_scores = stats.zscore(data1.select_dtypes(include=['number']))
# abs_z_scores = np.abs(z_scores)
# outliers = (abs_z_scores > 3).any(axis=1)
# data1_outliers = data1[outliers]
# print(data1_outliers)
# def whisker(col):
#     q1,q3 = np.percentile(col,[25,75])
#     iqr = q3-q1
#     lw = q1 - 1.5 * iqr
#     uw = q1 + 1.5 * iqr
#     return lw,uw
# for i in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
#     lw,uw = whisker(data1[i])
#     print(f'\t\t\tOutliers\n*****************************************************************************************')
#     print(f'{i}-lower:{lw}\t\t{i}-upper:{uw}\n')
#     # outliers = ((data1[i] < (lw)) | (data1[i] > (uw)))
#     # data1_outliers = data1[outliers]
#     # print(data1_outliers)
# #box plot to identify the distribution and outliers
# import seaborn as sns
# from matplotlib import pyplot as plt
# for i in data1.select_dtypes(include = ['number']).columns:
#     sns.boxplot(data = data1,x=i)
#     plt.show()
# from scipy import stats
# import numpy as np
# z_scores = stats.zscore(data2.select_dtypes(include=['number']))
# abs_z_scores = np.abs(z_scores)
# outliers = (abs_z_scores > 3).any(axis=1)
# data2_outliers = data2[outliers]
# print(data2_outliers)
# for i in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
#     lw,uw = whisker(data2[i])
#     print(f'\t\t\tOutliers\n*****************************************************************************************')
#     print(f'{i}-lower:{lw}\t\t{i}-upper:{uw}\n')
#     # outliers = ((data1[i] < (lw)) | (data1[i] > (uw)))
#     # data1_outliers = data1[outliers]
#     # print(data1_outliers)
# for i in data2.select_dtypes(include = 'number').columns:
#     sns.boxplot(data = data2,x=i)
#     plt.show()
# #scatter plot to understand the realationship between features and label
# for i in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
#     sns.scatterplot(data = data1,x = i,y = 'label')
#     plt.show()
# for i in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
#     sns.scatterplot(data = data2,x = i,y = 'label')
#     plt.show()
# #correlation with heatmap to interpret the relation and multicolliniarity
# set1 = data1.select_dtypes(include = 'number').corr()
# set1
# plt.figure(figsize = (10,8))
# sns.heatmap(set1,annot=True,fmt='.2f',cmap = 'coolwarm')
# plt.title('correlation matrix')
# set2 = data2.select_dtypes(include = 'number').corr()
# set2
# plt.figure(figsize = (10,8))
# sns.heatmap(set2,annot=True,fmt='.2f',cmap = 'coolwarm')
# plt.title('correlation matrix')
# data1.select_dtypes(include = 'number').columns
# data2.select_dtypes(include = 'number').columns


# import pandas as pd

# df1 = pd.read_csv(r'C:\Youtube\DiseaseDetection\dataset\crop_data1.csv')
# df1.head()
# import pandas as pd

# df2 = pd.read_csv(r'C:\Youtube\DiseaseDetection\dataset\crop_data2.csv')
# df2.head()
# display(df1.shape,df2.shape)
# merged_data = pd.concat([df1,df2],axis = 0)
# merged_data.shape
# merged_data
# # checking fro duplicated values
# merged_data.duplicated().sum()
# # checking for null values
# merged_data.isnull().sum()
# # Removing duplicates
# merged_data.drop_duplicates(subset = ['N','P','K','temperature','humidity','ph','rainfall'],keep = 'first',inplace = True)
# merged_data.duplicated().sum()
# final_dataset = merged_data
# final_dataset

# # Importing necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.naive_bayes import GaussianNB
# from sklearn import metrics
# #load the datasets
# data = final_dataset
# data
# data.head()
# data.info()
# data.describe()
# data.shape
# data.duplicated().sum()
# # Selecting continuous numerical columns and the label
# features = data.select_dtypes(include=[float, int]).columns
# target = 'label'
# # Splitting dataset as features and label
# X = data[features]
# y = data[target]
# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Scaling the dataset using standard scaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # X_train_scaled = pd.DataFrame(X_train_scaled, columns = features)
# # X_test_scaled = pd.DataFrame(X_test_scaled, columns = features)
# display(X_train_scaled,X_test_scaled)
# # Model Initializing using Gaussian NaiveBayes algorithm
# model = GaussianNB()
# model.fit(X_train_scaled, y_train)
# # Making predictions
# y_pred = model.predict(X_test_scaled)
# # Function to predict crop using scaeled data 
# def predict_crop(features):
#     features_df = pd.DataFrame([features], columns=features.keys())
#     scaled_features = scaler.transform(features_df)
#     prediction = model.predict(scaled_features)
#     return prediction[0]
# #input features
# features = {
#     'N': input('enter Nitrogen value:'),
#     'P': input('Enter Phosphorous values:'),
#     'K': input('Enter Potassium value:'),
#     'temperature': input('Enter the Temperature:'),
#     'humidity': input('Enter the Humidity:'),
#     'ph': input('Enter the pH value:'),
#     'rainfall': input('Enter the Rainfall')
# }
# print('Naive Bayes Prediction:', predict_crop(features))

# #importing necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load the dataset
# df = merged_data

# # Selecting continuous columns and the label
# continuous_columns = df.select_dtypes(include=[float, int]).columns
# label_column = 'label'

# # Splitting features and label
# X = df[continuous_columns]
# y = df[label_column]

# # Splitting the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initializing the scaler
# scaler = StandardScaler()

# # Fitting the scaler on the training data and transform both training and test data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # revertng the scaled array back to DataFrame to retain feature names
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=continuous_columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=continuous_columns)

# # Model initializing using Gaussian Naive Bayes algorithm
# model = GaussianNB()
# model.fit(X_train_scaled, y_train)

# # Predicting using the test data
# y_train_pred = model.predict(X_train_scaled)
# y_test_pred = model.predict(X_test_scaled)

# # Calculating accuracy
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# print(f"Training Accuracy: {train_accuracy*100}")
# print(f"Test Accuracy: {test_accuracy*100}")

# # Plot learning curves
# train_sizes = np.linspace(0.1, 1.0, 10, endpoint=False)
# train_scores = []
# test_scores = []

# for train_size in train_sizes:
#     # Using only  portion of the training data
#     X_train_partial, _, y_train_partial, _ = train_test_split(X_train_scaled, y_train, train_size=train_size, random_state=42)
#     model.fit(X_train_partial, y_train_partial)
#     train_scores.append(model.score(X_train_partial, y_train_partial))
#     test_scores.append(model.score(X_test_scaled, y_test))

# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores, label='Training Score')
# plt.plot(train_sizes, test_scores, label='Validation Score')
# plt.xlabel('Training Size')
# plt.ylabel('Accuracy')
# plt.title('Learning Curves')
# plt.legend()
# plt.show()
# from sklearn.metrics import accuracy_score

# # Training the model on training data
# model.fit(X_train_scaled, y_train)

# # Evaluating the model on training data
# train_predictions = model.predict(X_train_scaled)
# nb_train_accuracy = accuracy_score(y_train, train_predictions)
# print(f'Training Accuracy: {nb_train_accuracy*100}')

# # Evaluating the model on testing data
# test_predictions = model.predict(X_test_scaled)
# nb_test_accuracy = accuracy_score(y_test, test_predictions)
# print(f'Test Accuracy: {nb_test_accuracy*100}')
# from sklearn.model_selection import cross_val_score

# # Performing cross-validation for model optimization
# cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
# print(f'Cross-Validation Scores: {cv_scores*100}')
# print(f'Mean CV Score: {cv_scores.mean()*100}')
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import learning_curve

# train_sizes, train_scores, test_scores = learning_curve(model, X_train_scaled, y_train, cv=5,
#                                                         train_sizes=np.linspace(0.1, 1.0, 10),
#                                                         scoring='accuracy')

# # Calculating mean and standard deviation
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
# plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1, color='r')
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color='g')

# plt.xlabel('Training Size')
# plt.ylabel('Score')
# plt.title('Learning Curves')
# plt.legend(loc='best')
# plt.show()
# # Model Evaluation
# from sklearn import metrics
# accuracy = metrics.accuracy_score(y_test, y_pred)
# precision = metrics.precision_score(y_test,y_pred,average = 'weighted',zero_division=0)
# recall = metrics.recall_score(y_test,y_pred,average = 'weighted')
# f1 = metrics.f1_score(y_test,y_pred,average = 'weighted')
# print(f'Model Accuracy: {accuracy * 100:.2f}%')
# print(f'Precision Score: {precision * 100:.2f}%')
# print(f'Recall Score: {recall * 100: .2f}%')
# print(f'f1-score: {f1*100:.2f}%')
# cm = metrics.confusion_matrix(y_test,y_pred)
# print(f'Confusion matrix:\n{cm}')
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()
# cr = metrics.classification_report(y_test,y_pred,zero_division=0)
# print(f'Classification Report:\n{cr}')

# #training of data and checking libraries with more accuracy 
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import ExtraTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import accuracy_score


# #libraries used in dictionary format
# models = {
#     'Logistic Regression' : LogisticRegression(),
#     'Naive Bayes' : GaussianNB(),
#     'Support Vector Machine' : SVC(),
#     'K-Nearest Neighbors' : KNeighborsClassifier(),
#     'Decision Tree' : DecisionTreeClassifier(),
#     'Random Forest' : RandomForestClassifier(),
#     'Bagging' : BaggingClassifier(),
#     'AdaBoost' : AdaBoostClassifier(),
#     'Gradient Boosting' : GradientBoostingClassifier(),
#     'Extra Trees': ExtraTreeClassifier(),
# }

# #name = libraries/modulesnames, md = models dictionary and results will be assigned in items
# for name,md in models.items():
#     md.fit(X_train_scaled,y_train)
#     ypred = md.predict(X_test_scaled)

#     print(f"{name} with accuracy : {accuracy_score(y_test,ypred) * 100}")

# # Importing necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn import metrics
# # Load the dataset
# data = final_dataset
# # Selecting continuous numerical columns and the label
# features = data.select_dtypes(include=[float, int]).columns
# target = 'label'
# # Splitting dataset into features and label
# X = data[features]
# y = data[target]
# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Scaling the dataset using standard scaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # model initializing using svm classifier
# model = SVC(kernel='linear', random_state=42)  
# model.fit(X_train_scaled, y_train)
# # Making predictions
# y_pred = model.predict(X_test_scaled)
# # Function to predict crop recommendation 
# def predict_crop(features):
#     features_df = pd.DataFrame([features], columns=features.keys())
#     scaled_features = scaler.transform(features_df)
#     prediction = model.predict(scaled_features)
#     return prediction[0]
# #input features
# features = {
#     'N': input('enter Nitrogen value:'),
#     'P': input('Enter Phosphorous values:'),
#     'K': input('Enter Potassium value:'),
#     'temperature': input('Enter the Temperature:'),
#     'humidity': input('Enter the Humidity:'),
#     'ph': input('Enter the pH value:'),
#     'rainfall': input('Enter the Rainfall')
# }
# print('Naive Bayes Prediction:', predict_crop(features))

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load the dataset
# df = merged_data

# # Selecting continuous columns and the label
# continuous_columns = df.select_dtypes(include=[float, int]).columns
# label_column = 'label'

# # Splitting features and label
# X = df[continuous_columns]
# y = df[label_column]

# # Splitting the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initializing the scaler
# scaler = StandardScaler()

# # Fitting the scaler on the training data and transform both training and test data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# # reverting back to DataFrame to retain feature names
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=continuous_columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=continuous_columns)

# # Initializing and train the Naive Bayes Classifier
# model = SVC()
# model.fit(X_train_scaled, y_train)

# # Predicting using the test data
# y_train_pred = model.predict(X_train_scaled)
# y_test_pred = model.predict(X_test_scaled)

# # Calculating accuracy
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# print(f"Training Accuracy: {train_accuracy*100}")
# print(f"Test Accuracy: {test_accuracy*100}")

# # Plottting learning curves
# train_sizes = np.linspace(0.1, 1.0, 10, endpoint=False)
# train_scores = []
# test_scores = []

# for train_size in train_sizes:
#     # Using a portion of the training data
#     X_train_partial, _, y_train_partial, _ = train_test_split(X_train_scaled, y_train, train_size=train_size, random_state=42)
#     model.fit(X_train_partial, y_train_partial)
#     train_scores.append(model.score(X_train_partial, y_train_partial))
#     test_scores.append(model.score(X_test_scaled, y_test))

# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores, label='Training Score')
# plt.plot(train_sizes, test_scores, label='Validation Score')
# plt.xlabel('Training Size')
# plt.ylabel('Accuracy')
# plt.title('Learning Curves')
# plt.legend()
# plt.show()
# from sklearn.metrics import accuracy_score

# # Training the model on training data
# model.fit(X_train_scaled, y_train)

# # Evaluating the model on training data
# train_predictions =model.predict(X_train_scaled)
# svm_train_accuracy = accuracy_score(y_train, train_predictions)
# print(f'Training Accuracy: {svm_train_accuracy*100}')

# # Evaluating the model on test data
# test_predictions = model.predict(X_test_scaled)
# svm_test_accuracy = accuracy_score(y_test, test_predictions)
# print(f'Test Accuracy: {svm_test_accuracy*100}')
# from sklearn.model_selection import cross_val_score

# # Performing cross-validation
# cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
# print(f'Cross-Validation Scores: {cv_scores*100}')
# print(f'Mean CV Score: {cv_scores.mean()*100}')
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import learning_curve

# train_sizes, train_scores, test_scores = learning_curve(model, X_train_scaled, y_train, cv=5,
#                                                         train_sizes=np.linspace(0.1, 1.0, 10),
#                                                         scoring='accuracy')

# # Calculating mean and standard deviation
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
# plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1, color='r')
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color='g')

# plt.xlabel('Training Size')
# plt.ylabel('Score')
# plt.title('Learning Curves')
# plt.legend(loc='best')
# plt.show()
# # Model Evaluation
# from sklearn import metrics
# accuracy = metrics.accuracy_score(y_test, y_pred)
# precision = metrics.precision_score(y_test,y_pred,average = 'weighted',zero_division=0)
# recall = metrics.recall_score(y_test,y_pred,average = 'weighted')
# f1 = metrics.f1_score(y_test,y_pred,average = 'weighted')
# print(f'Model Accuracy: {accuracy * 100:.2f}%')
# print(f'Precision Score: {precision * 100:.2f}%')
# print(f'Recall Score: {recall * 100: .2f}%')
# print(f'f1-score: {f1*100:.2f}%')
# cm = metrics.confusion_matrix(y_test,y_pred)
# print(f'Confusion matrix:\n{cm}')
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()
# cr = metrics.classification_report(y_test,y_pred,zero_division=0)
# print(f'Classification Report:\n{cr}')
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load the dataset
# df = merged_data

# # Select continuous columns and the label
# continuous_columns = df.select_dtypes(include=[float, int]).columns
# label_column = 'label'

# # Split features and label
# X = df[continuous_columns]
# y = df[label_column]

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the scaler
# scaler = StandardScaler()

# # Fit the scaler on the training data and transform both training and test data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # Convert back to DataFrame to retain feature names
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=continuous_columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=continuous_columns)

# # Initialize and train the SVM Classifier
# svm_clf = SVC(kernel='linear', random_state=42)
# svm_clf.fit(X_train_scaled, y_train)

# # Predict using the SVM model
# y_train_pred_svm = svm_clf.predict(X_train_scaled)
# y_test_pred_svm = svm_clf.predict(X_test_scaled)

# # Calculate SVM accuracy
# train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
# test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)

# print(f"SVM Training Accuracy: {train_accuracy_svm}")
# print(f"SVM Test Accuracy: {test_accuracy_svm}")

# # Initialize and train the Naive Bayes Classifier
# nb_clf = GaussianNB()
# nb_clf.fit(X_train_scaled, y_train)

# # Predict using the Naive Bayes model
# y_train_pred_nb = nb_clf.predict(X_train_scaled)
# y_test_pred_nb = nb_clf.predict(X_test_scaled)

# # Calculate Naive Bayes accuracy
# train_accuracy_nb = accuracy_score(y_train, y_train_pred_nb)
# test_accuracy_nb = accuracy_score(y_test, y_test_pred_nb)

# print(f"Naive Bayes Training Accuracy: {train_accuracy_nb}")
# print(f"Naive Bayes Test Accuracy: {test_accuracy_nb}")

# # Plotting the accuracies
# labels = ['SVM', 'Naive Bayes']
# train_accuracies = [train_accuracy_svm, train_accuracy_nb]
# test_accuracies = [test_accuracy_svm, test_accuracy_nb]

# x = np.arange(len(labels))  # the label locations
# width = 0.35  

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, train_accuracies, width, label='Training Accuracy')
# rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Model')
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy Comparison between SVM and Naive Bayes')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # Attach a text label above each bar in *rects*, displaying its height.
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(round(height, 2)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

# fig.tight_layout()
# plt.show()
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load the dataset
# df = merged_data

# # Select continuous columns and the label
# continuous_columns = df.select_dtypes(include=[float, int]).columns
# label_column = 'label'

# # Split features and label
# X = df[continuous_columns]
# y = df[label_column]

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the scaler
# scaler = StandardScaler()

# # Fit the scaler on the training data and transform both training and test data
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # Convert back to DataFrame to retain feature names
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=continuous_columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=continuous_columns)

# # Initialize and train the SVM Classifier
# svm_clf = SVC(kernel='linear', random_state=42)
# svm_clf.fit(X_train_scaled, y_train)

# # Predict using the SVM model
# y_train_pred_svm = svm_clf.predict(X_train_scaled)
# y_test_pred_svm = svm_clf.predict(X_test_scaled)

# # Calculate SVM accuracy
# train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
# test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)

# print(f"SVM Training Accuracy: {train_accuracy_svm}")
# print(f"SVM Test Accuracy: {test_accuracy_svm}")

# # Initialize and train the Naive Bayes Classifier
# nb_clf = GaussianNB()
# nb_clf.fit(X_train_scaled, y_train)

# # Predict using the Naive Bayes model
# y_train_pred_nb = nb_clf.predict(X_train_scaled)
# y_test_pred_nb = nb_clf.predict(X_test_scaled)

# # Calculate Naive Bayes accuracy
# train_accuracy_nb = accuracy_score(y_train, y_train_pred_nb)
# test_accuracy_nb = accuracy_score(y_test, y_test_pred_nb)

# print(f"Naive Bayes Training Accuracy: {train_accuracy_nb}")
# print(f"Naive Bayes Test Accuracy: {test_accuracy_nb}")

# # Plotting the accuracies as points
# models = ['SVM', 'Naive Bayes']
# train_accuracies = [train_accuracy_svm, train_accuracy_nb]
# test_accuracies = [test_accuracy_svm, test_accuracy_nb]

# # Plotting the accuracies
# plt.figure(figsize=(10, 6))
# plt.plot(models, train_accuracies, 'o-', label='Training Accuracy', color='blue')
# plt.plot(models, test_accuracies, 'o-', label='Test Accuracy', color='red')
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Comparison between SVM and Naive Bayes')
# plt.legend()
# plt.grid(True)
# plt.show()
# # Bar chart for accuracies
# fig, ax = plt.subplots(figsize=(10, 6))
# x = np.arange(len(models))
# width = 0.35

# rects1 = ax.bar(x - width/2, train_accuracies, width, label='Training Accuracy', color='blue')
# rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='red')

# ax.set_xlabel('Model')
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy Comparison between SVM and Naive Bayes')
# ax.set_xticks(x)
# ax.set_xticklabels(models)
# ax.legend()

# # Attach a text label above each bar
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)

# fig.tight_layout()
# plt.show()


# train_model.py
"""
Train crop recommendation model using Kaggle dataset Crop_recommendation.csv
Trains and compares: RandomForest, KNN, SVM, DecisionTree (only these four).
Saves the best pipeline to best_crop_pipeline.joblib and label encoder to label_encoder.joblib.

Usage:
    python train_model.py --data_path ./Crop_recommendation.csv --use_smote False
"""

import warnings, argparse, os
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def normalize_labels(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    # merge trivial plural variants like 'groundnuts' -> 'groundnut'
    # and remove extra punctuation/whitespace
    s = s.str.replace(r'[^a-z0-9]', '', regex=True)
    # map plurals that become duplicates (simple heuristic): if both 'x' and 'xs' present map 'xs' -> 'x'
    vals = set(s.unique())
    mapping = {}
    for v in list(vals):
        if v.endswith('s'):
            singular = v[:-1]
            if singular in vals:
                mapping[v] = singular
    if mapping:
        s = s.map(lambda x: mapping.get(x, x))
    return s

def main(data_path, use_smote=False, verbose=True):
    assert os.path.exists(data_path), f"data_path not found: {data_path}"
    df = pd.read_csv(data_path)
    # ensure expected columns exist
    missing_cols = [c for c in FEATURES + ['label'] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Drop exact duplicates rows (on numeric features + label)
    df = df.drop_duplicates(subset=FEATURES + ['label']).reset_index(drop=True)
    if verbose:
        print("Loaded dataset shape:", df.shape)

    # Normalize label text (lowercase, strip, remove punctuation)
    df['label_norm'] = normalize_labels(df['label'])
    print("Unique labels before:", df['label'].nunique(), "after normalization:", df['label_norm'].nunique())

    # Optional: manual corrections (if you know a few mappings). Example:
    # df['label_norm'] = df['label_norm'].replace({'groundnuts':'groundnut'})

    X = df[FEATURES].astype(float)
    y_raw = df['label_norm']

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    print("Classes (sample):", class_names[:10], "... total:", len(class_names))

    # Save encoder for later (Flask app uses this)
    joblib.dump(le, "label_encoder.joblib")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print("Train / Test shapes:", X_train.shape, X_test.shape)

    # Optional: SMOTE to reduce class imbalance (only on train)
    if use_smote:
        sm = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print("After SMOTE train shape:", X_train_res.shape)
    else:
        X_train_res, y_train_res = X_train.values, y_train

    # Define pipelines (scale + classifier)
    pipelines = {
        "RandomForest": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'))]),
        "DecisionTree": Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'))]),
    }

    param_grids = {
        "RandomForest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 20], "clf__min_samples_leaf": [1,2]},
        "KNN": {"clf__n_neighbors": [3,5,7], "clf__weights": ['uniform','distance']},
        "SVM": {"clf__C": [1,10], "clf__kernel": ['rbf'], "clf__gamma": ['scale']},
        "DecisionTree": {"clf__max_depth": [None, 20], "clf__min_samples_leaf": [1,2]},
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_pipelines = {}
    test_scores = {}

    # Train each model with GridSearchCV
    for name, pipe in pipelines.items():
        print(f"\nTraining {name} ...")
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
        grid.fit(X_train_res, y_train_res)
        best = grid.best_estimator_
        best_pipelines[name] = best
        # Evaluate on hold-out test
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_scores[name] = acc
        print(f"  Best CV score: {grid.best_score_:.4f}; Test accuracy: {acc:.4f}; Best params: {grid.best_params_}")

    # Select best by test accuracy
    best_name = max(test_scores, key=test_scores.get)
    best_pipeline = best_pipelines[best_name]
    print(f"\nSelected best model: {best_name} with test acc {test_scores[best_name]:.4f}")

    # Save the final pipeline
    joblib.dump(best_pipeline, "best_crop_pipeline.joblib")
    print("Saved pipeline -> best_crop_pipeline.joblib and label_encoder.joblib")

    # Detailed evaluation
    y_pred_best = best_pipeline.predict(X_test)
    print("\nClassification report (best model):")
    print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Utility: top-k predictions
    def predict_topk(features: dict, k=3):
        # features must have FEATURES keys
        X_in = pd.DataFrame([features])[FEATURES].astype(float)
        if hasattr(best_pipeline.named_steps['clf'], "predict_proba"):
            probs = best_pipeline.predict_proba(X_in)[0]
            topk_idx = np.argsort(probs)[::-1][:k]
            return [(class_names[i], float(probs[i])) for i in topk_idx]
        else:
            pred = best_pipeline.predict(X_in)[0]
            return [(class_names[pred], 1.0)]

    # Save a small demo - just for user to try with an example row
    example_row = X_test.iloc[0].to_dict()
    print("\nExample test row:", example_row)
    print("Top-3 predictions for example row:", predict_topk(example_row, k=3))

    return best_name, best_pipeline, le

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"C:\Youtube\DiseaseDetection\dataset\Crop_recommendation.csv")
    parser.add_argument("--use_smote", type=lambda x: x.lower() in ("1","true","yes"), default=False)
    args = parser.parse_args()
    main(args.data_path, use_smote=args.use_smote)
