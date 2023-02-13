# Text_preprocessing
Text_preprocessing in python

!pip install snscrape

#Defining Function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
df['cleaned_tweets']= df['positive_tweets'].apply(lambda x:remove_punctuation(x))
display(df.head(3))

#Lowering the tweets
df['lower_tweets']= df['cleaned_tweets'].apply(lambda x: x.lower())
df

#Tokenization
from nltk.tokenize import TweetTokenizer as tt
#applying function to the column
tokenizer = tt()   
df['tok_tweets'] = df['lower_tweets'].apply(lambda x: tokenizer.tokenize(x))
df

#Removing stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stopword = stopwords.words('english')

#Defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopword]
    return output
#applying the function
df['no_stopwords']= df['tok_tweets'].apply(lambda x:remove_stopwords(x))
df

#Stemming 
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

#Defining a function for stemming
def stemming(text):
    stem_tweet = [porter_stemmer.stem(word) for word in text]
    return stem_tweet
df['stem_tweets']=df['no_stopwords'].apply(lambda x: stemming(x))
df

#Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemma = WordNetLemmatizer()

#Defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [lemma.lemmatize(word) for word in text]
    return lemm_text
df['lemmatized_tweets']=df['stem_tweets'].apply(lambda x:lemmatizer(x))
df.head(5)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections

l1=df.lemmatized_tweets.sum()
c1=[a for a, v in collections.Counter(l1).items() if v > 40]

cloud1 = WordCloud(width =700, height = 400).generate((" ").join(c1))
plt.figure(figsize=(15,5))
print(plt.imshow(cloud1))
plt.axis("off")
plt.show()
plt.close()
