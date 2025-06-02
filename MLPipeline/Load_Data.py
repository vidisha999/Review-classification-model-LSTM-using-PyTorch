import pandas as pd

class Load_Data:

    def load_data(self):
        data = pd.read_csv('Data/review_data_LSTM.csv')
        data = data[['content', 'score']] # select necessary columns
        return data
