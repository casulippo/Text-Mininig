import pandas as pd
import os
import numpy as np
import seaborn as sns
import re
import gzip
import json
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import unicodedata
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
import contractions
import re
import inflect
from keras.layers import Dense, Dropout,Embedding, LSTM, GRU
from keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import TruncatedSVD

#Word Cloud
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image
from collections import Counter

#Text Representation
from sklearn.feature_extraction.text import CountVectorizer

#Text Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix

#Visualization Word Embedding 
import plotly.offline as py
from sklearn.manifold import TSNE
import plotly.graph_objs as go

#Sentiment Analysis
nltk.download('opinion_lexicon')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn

#Text Clustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import normalized_mutual_info_score 

#LDA (Topic Modeling)
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from pprint import pprint
from gensim.models import CoherenceModel
