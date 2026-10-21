# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:37:56 2026

@author: Kali
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')



nltk.download('stopwords')


data = pd.read_csv('Emails.csv')
print(data.head())


sns.countplot(x='label', data=data)
plt.show()

ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']

# Downsample Ham emails to match the number of Spam emails
ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)


# Combine balanced data
balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)

sns.countplot(x='label', data=balanced_data)
plt.show()



balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
balanced_data.head()



punctuations_list = string.punctuation
def remove_punk(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

balanced_data['text']= balanced_data['text'].apply(lambda x: remove_punk(x))

balanced_data.head()


def remove_stopords(txt):
    stop_words = stopwords.words('english')
    
    imp_words = []
    
    #storing the important words
    for word in str(txt).split():
        word = word.lower()
        
        if word not in stop_words:
            imp_words.append(word)
            
    output = " ".join(imp_words)
    
    return output


balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopords(text))

balanced_data.head()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        



