import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB

# Read the raw data with error handling
try:
    raw_data = pd.read_csv('spam.csv', encoding='latin1')

except FileNotFoundError:
    print("File 'spam.csv' not found!")
    exit(1)

# Display the first few rows of the raw data 
# This line displays the first few rows of the raw data to get an idea of what the data looks like
raw_data.head()

# Check value counts of the 'Category' column
# There are 4825 occurrences of 'ham' and 747 occurrences of 'spam' in the 'Category' column
raw_data['Category'].value_counts()

# Delete rows with '{"mode":"full"' as their category
# Clean up the dataset by removing anomalous entries (except for spam and ham)
data = raw_data[raw_data['Category'] != '{"mode":"full"']

# Delete duplicate rows
data = data.drop_duplicates()

# Map 'ham' to 0 and 'spam' to 1 in the 'Category' column
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Shuffle the dataset
data = shuffle(data, random_state=20)

# Balance the dataset by undersampling the majority class (ham)
# Selects a random subset of ham samples equal in size to the number of spam samples to create a balanced dataset
spam_count = data['Category'].sum()
ham_indices = data[data['Category'] == 0].index
random_indices = np.random.choice(ham_indices, spam_count, replace=False)
spam_indices = data[data['Category'] == 1].index
under_sample_indices = np.concatenate([spam_indices, random_indices])
balanced_data = data.loc[under_sample_indices]

# Split data into features (x) and target variable (y)
x = balanced_data['Message']
y = balanced_data['Category']

# Tokenize the 'Message' column using CountVectorizer
# CountVectorizer is used to tokenize the text data in the 'Message' column. It converts each message into a numerical representation suitable for machine learning algorithms
# fit_transform() method fits the CountVectorizer to the text data (x) and transforms it into a matrix of token counts
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split the dataset into training and testing sets
# 0.2 (20%) of the data is spilt as a test data
# Balance 80% is a train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(x_train, y_train)

# Predictions on the training set
y_pred_train = model.predict(x_train)

# Model evaluation on the training set
accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
recall_train = metrics.recall_score(y_train, y_pred_train)
precision_train = metrics.precision_score(y_train, y_pred_train)

print('Training Set Metrics:')
print('Accuracy:', accuracy_train)
print('Recall:', recall_train)
print('Precision:', precision_train)

# Confusion matrix for the training set
cf_matrix_train = metrics.confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(4, 3))
sns.heatmap(cf_matrix_train, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['True Ham', 'True Spam'])
plt.title('Confusion Matrix For Training Data')
plt.show()

# Predictions on the test set
y_pred_test = model.predict(x_test)

# Model evaluation on the test set

accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
# Calculates the accuracy of the model's predictions (y_pred_train) compared to the actual target labels (y_train) in the training set

recall_test = metrics.recall_score(y_test, y_pred_test)
# Calculates the recall (also known as sensitivity or true positive rate) of the model's predictions (y_pred_train) compared to the actual target labels (y_train) in the training set

precision_test = metrics.precision_score(y_test, y_pred_test)
# Calculates the precision of the model's predictions (y_pred_train) compared to the actual target labels (y_train) in the training set.
# Precision measures the proportion of predicted positive cases that were truly positive.

#Print Values
print('Test Set Metrics:')
print('Accuracy:', accuracy_test)
print('Recall:', recall_test)
print('Precision:', precision_test)

# Confusion matrix is a table that summarizes the performance of a classification algorithm
# Contains information about the true positive, false positive, true negative, and false negative predictions
cf_matrix_test = metrics.confusion_matrix(y_test, y_pred_test)

# Creates figure with a specific size (4 inches wide and 3 inches tall)
plt.figure(figsize=(4, 3))

# From the Seaborn library creates a heatmap visualization of the confusion matrix
# cf_matrix_test passed as the data for the heatmap
# annot=True adds numeric annotations (the count of samples) to each cell of the heatmap
# cmap = color
# fmt='d' specifies the format of the annotations as integers
# cbar=False removes the color bar from the side of the heatmap
sns.heatmap(cf_matrix_test, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['Predicted Ham', 'Predicted Spam'],
            yticklabels=['True Ham', 'True Spam'])
plt.title('Confusion Matrix For Test Data')
plt.show()
