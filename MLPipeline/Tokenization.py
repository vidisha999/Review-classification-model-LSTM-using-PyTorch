
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Tokenization:

    def generate_token(self,data1):
        top_length = 2000  # The maximum number of words to be used(vocabulary length)
        input_length = 600 # Max number of words in each content
        embedding_dim = 100 # Size of word vector embeddings
        tokenizer = Tokenizer(num_words=top_length, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(data1['content'].values)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        
        # Tokenizing the content
        X = tokenizer.texts_to_sequences(data1['content'].values)
        X = pad_sequences(X, maxlen=input_length)
        return word_index, X