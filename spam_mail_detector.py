# Load, explore and plot data
import numpy as np
import pandas as pd
import re
import string
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
# suppress display of warnings
import warnings
warnings.filterwarnings("ignore")

## Reading the given dataset
data = pd.read_csv("SMSSpamCollection.txt", sep = "\t", names=["label", "message"])

data.head()

#data.shape

# Checking for NUll values
data.isnull().sum()

# Checking for duplicate values
data.duplicated().sum()

# drop duplicates
data = data.drop_duplicates()

data.duplicated().sum()

#data.shape

data['label'].value_counts()

plt.figure(figsize=(9, 5))

plt.pie(data['label'].value_counts(),labels=['ham','spam'],autopct='%0.2f', colors=['Green', 'Red'], explode = [.1, .1])
plt.show()

#data['message'][0]

#natural language tool kit
import nltk
nltk.download('punkt')

# For a number of characters

data['num_characters']=data['message'].apply(len)
data.head()

# For a number of words
from nltk.tokenize import word_tokenize
data['message'].apply(lambda x: nltk.word_tokenize(x))

data['num_words']=data['message'].apply(lambda x:len(nltk.word_tokenize(x)))
data.head()

# For a number of sentences
data['num_sentences']=data['message'].apply(lambda x: len(nltk.sent_tokenize(x)))

data.head()

# For Ham messages
data[data['label']=='ham'][['num_characters','num_words','num_sentences']].describe()

# For Spam messages:
data[data['label']=='spam'][['num_characters','num_words','num_sentences']].describe()

#for characters
plt.figure(figsize=(12,6))
sns.histplot(data[data['label']=='ham']['num_characters'],color='green')
sns.histplot(data[data['label']=='spam']['num_characters'],color = 'red')

#for words
plt.figure(figsize=(12,6))
sns.histplot(data[data['label']=='ham']['num_words'],color='green')
sns.histplot(data[data['label']=='spam']['num_words'],color='red')

from wordcloud import WordCloud

ham_msg_text = data[data.label == 'ham'].message
spam_msg_text = data[data.label == 'spam'].message

## Word Cloud for Ham MSGs
plt.figure(figsize = (20, 22))
wc = WordCloud(width = 1500, height = 900, max_words = 2500).generate(" ".join(ham_msg_text))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()       # Display the figure
plt.close()
## Word Cloud for Spam MSGs
plt.figure(figsize = (20, 22))
wc = WordCloud(width = 1500, height = 900, max_words = 2500).generate(" ".join(spam_msg_text))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()       # Display the figure
plt.close()

u = data['message'][3]
#u

v = data['message'][8]
#v

def remove_punc(text):
  trans = str.maketrans('', '', string.punctuation)
  return text.translate(trans)

data['message'] = data['message'].apply(remove_punc)

#data['message'][8]

def remove_noise(text):
  t = re.sub('[^a-zA-Z]', ' ', text)
  return t

data['message'] = data['message'].apply(remove_noise)

#data['message'][8]

nltk.download('stopwords')

from nltk.corpus import stopwords
sw = stopwords.words('english')

len(sw)

def remove_sws(text):
  s = [word.lower() for word in text.split() if word.lower() not in sw]
  return " ".join(s)

data['message'] = data['message'].apply(remove_sws)

#data['message'][8]

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemma(text):
  l = [lemmatizer.lemmatize(word) for word in text.split()]
  return " ".join(l)

data['message'] = data['message'].apply(lemma)

data.head()

from sklearn.preprocessing import LabelEncoder
encoder =LabelEncoder()

data['label']=encoder.fit_transform(data['label'])

data.head()

# Select only the relevant columns for building model

data = data[['label','message']]

data.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=3000)

X = tf.fit_transform(data['message']).toarray()
y = data['label']

#X

#y

# Splitting the data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create the instance of Naive Bayes
clf = BernoulliNB()

# Fit the data
clf.fit(X_train, y_train)

# Making prediction
y_pred = clf.predict(X_test)

# Evaluation
#print("Accuracy Score: ", accuracy_score(y_test, y_pred))

#print(classification_report(y_test, y_pred))

# Plot the Confusion Matrix

import seaborn as sn
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (12,5))
sn.heatmap(cm, annot=True,linewidth = 0.5 , cmap = 'tab20' , fmt='d')
plt.title('Naive Bayes Model Confusion Matrix', size=15)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Streamlit app UI
st.title('Email Spam Detector')
    
user_input = st.text_input('Enter the email text:')
if st.button('Predict'):
    input_vector = tf.transform([user_input])
    prediction = clf.predict(input_vector)
    result = 'Spam' if prediction[0] == 1 else 'Ham'
    st.write(f'Prediction: {result}')