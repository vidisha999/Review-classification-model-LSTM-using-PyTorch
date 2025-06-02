
import re
import tensorflow as tf
import numpy as np
import nltk

nltk.download('stopwords')
from tensorflow import keras
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from MLPipeline.Load_Data import Load_Data
from MLPipeline.Preprocessing import Preprocessing
from MLPipeline.Tokenization import Tokenization
from MLPipeline.Create import Create
from MLPipeline.LSTM import LSTM
from MLPipeline.Training import Training

max_features = 2000
batch_size = 50
vocab_size = max_features

#Loading data
data = Load_Data().load_data()
print(data)

# Preprocess data
preprocess = Preprocessing()
data['content'] = data['content'].apply(preprocess.clean_text)  # apply the function to every text in the dataset

#Resampling the imbalanced dataset.
data1 = preprocess.sampling(data)

#Tokeninzation
word_index, X = Tokenization().generate_token(data1)

#Encoding Labels
le, Y = preprocess.encoder(data1)

#create dataset
X_train, X_test, Y_train, Y_test = Create().create_dataset(X, Y)
x_cv, y_cv, train_dl, val_dl = Create().data_loader(X_train, X_val, Y_train, Y_val)

#define and instantiate model
model = LSTM(vocab_size, 128, 64)
print(model)

# Train the model
Training().train_val(10, model, train_dl, x_cv, val_dl, Y_val)
