import numpy as np
import pandas as pd
from keras.models import load_model

encoding_df = pd.read_csv(
    '/Users/subir/Codes/Miscellaneous/AnalytixLab/Mixed - Digital - Text Classification-Python/encodings.csv')


class Model:

    def __init__(self):
        pass

    def pred_using_bilstm(self, ip_seq):
        model = load_model(
            "/Users/subir/Codes/Miscellaneous/AnalytixLab/Mixed - Digital - Text Classification-Python/ModelCheckpoint/BiLSTM_Fasttext.23-0.711.hdf5")

        encoding = np.argmax(model.predict(ip_seq.reshape(1, -1)))

        pred_label = encoding_df.loc[encoding_df.encoding == encoding, 'label'].values[0]

        return pred_label
