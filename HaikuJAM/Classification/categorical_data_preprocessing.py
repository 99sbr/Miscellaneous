import pandas as pd
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(n_features=200)


# summarize encoded vector
# print(vector.shape)
# print(vector.toarray())
#
# #


def text_cleaning(data, column_name):
    # function to remove non-ascii characters
    def _removeNonAscii(s): return "".join(i.lower()
                                           for i in s if ord(i) < 128)

    # remove non-ascii characters
    data[column_name] = data[column_name].map(
        lambda x: _removeNonAscii(str(x)))
    data[column_name] = data[column_name].str.replace("[^a-zA-Z0-9]", " ")
    return data[column_name]


def preprocess_categorical_df(categorical_df):
    categorical_df.fillna("Not available", inplace=True)
    from tqdm import tqdm
    for column in tqdm(categorical_df.columns):
        categorical_df[column] = text_cleaning(categorical_df, column)

    categorical_df['Categorical_Description'] = categorical_df['Amenities'] + categorical_df['Summary'] + \
                                                categorical_df['Transit']

    categorical_df = categorical_df[
        ['Categorical_Description', 'Host_Response_Time', 'Calendar_Updated', 'Property_Type', 'Cancellation_Policy']]

    vectorizer = HashingVectorizer(n_features=100)
    vector = vectorizer.transform(categorical_df.Categorical_Description.values.tolist())
    embedding_matrix = vector.toarray()
    print('Shape of Embedding Matrix: ', embedding_matrix.shape)
    embed_df = pd.DataFrame(data=embedding_matrix)
    embed_df.columns = ['USE_' + str(i) for i in range(embedding_matrix.shape[1])]

    embed_df.Host_Response_Time = label_encoder.fit_transform(categorical_df.Host_Response_Time)
    embed_df.Calendar_Updated = label_encoder.fit_transform(categorical_df.Calendar_Updated)
    embed_df.Property_Type = label_encoder.fit_transform(categorical_df.Property_Type)
    embed_df.Cancellation_Policy = label_encoder.fit_transform(categorical_df.Cancellation_Policy)

    del categorical_df
    import gc
    print('GC Cleaned: ', gc.collect())
    return embed_df
