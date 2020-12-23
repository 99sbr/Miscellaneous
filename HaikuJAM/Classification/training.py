import gc
from nltk.corpus import stopwords
import pandas as pd
from categorical_data_preprocessing import preprocess_categorical_df
from numerical_data_preprocessing import preprocessing_numerics_df
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utility import null_values
import time
import warnings
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()

start = time.time()
warnings.filterwarnings("ignore")  # Ignoring unnecessory warnings


stop_words = stopwords.words('english')
#

print("------- LOADING DATA ------")
listings = pd.read_csv(
    '/Users/subir/Codes/Miscellaneous/HaikuJAM/listings.csv', delimiter=';')

print('Data Shape: ', listings.shape)
train_df = listings.drop(['Unnamed: 0', 'ID'], 1)
null_val_df = null_values(train_df)
columns_to_drop = null_val_df[null_val_df.Percent > 50].index.to_list()

for col in columns_to_drop:
    train_df.drop([col], 1, inplace=True)

numerical_cols = [
    'Host_Since', 'Host_Listings_Count', 'Accommodates', 'Cleaning_Fee',
    'Guests_Included', 'Maximum_Nights', 'Calendar_last_Scraped',
    'First_Review', 'Last_Review', 'Geolocation'
]

all_numerical_columns = train_df.select_dtypes(
    exclude=['object']).columns.to_list() + numerical_cols

numerical_df = train_df[all_numerical_columns]
print('Numerical Data Shape: ', numerical_df.shape)
print('Preprocessing Numerical Data in Progress')
numerical_df = preprocessing_numerics_df(numerical_df)

all_object_columns = list(
    set(train_df.columns.to_list()) - set(all_numerical_columns))
categorical_df = train_df[all_object_columns]
print('Categorical Data Shape: ', categorical_df.shape)
target = categorical_df.Listing_Type
categorical_df.drop('Listing_Type', 1, inplace=True)

print('Preprocessing Categorical Data in Progress')
categorical_df = preprocess_categorical_df(categorical_df)

assert categorical_df.shape[0] == numerical_df.shape[0]

print('Preprocessing Done')
head_df = categorical_df.join(numerical_df)
head_df['Listing_Type'] = target
head_df.loc[head_df.Listing_Type == 'Good', 'Listing_Type'] = 1
head_df.loc[head_df.Listing_Type == 'Bad', 'Listing_Type'] = 0
del categorical_df
del numerical_df


print('GC Cleaned: ', gc.collect())


def create_data_subsets(head_df):
    minority_df = head_df[head_df.Listing_Type == 0]
    majority_df = head_df[head_df.Listing_Type == 1]

    # creating 5 sub_heads from majority_df
    total_len = len(majority_df)
    sub_split_len = total_len // 5
    majority_df.sample(frac=1)
    majority_df1 = majority_df[:sub_split_len]
    majority_df2 = majority_df[sub_split_len:sub_split_len * 2]
    majority_df3 = majority_df[sub_split_len * 2:sub_split_len * 3]
    majority_df4 = majority_df[sub_split_len * 3:sub_split_len * 4]
    majority_df5 = majority_df[sub_split_len * 4:]
    print('Total dataset size: ', total_len)
    print('Subset shapes: ',
          (len(majority_df1), len(majority_df2), len(majority_df3), len(majority_df4), len(majority_df5)))

    # combine and shuffle
    df1 = majority_df1.append(minority_df)
    df2 = majority_df2.append(minority_df)
    df3 = majority_df3.append(minority_df)
    df4 = majority_df4.append(minority_df)
    df5 = majority_df5.append(minority_df)

    return df1.sample(frac=1), df2.sample(frac=1), df3.sample(frac=1), df4.sample(frac=1), df5.sample(frac=1)


df1, df2, df3, df4, df5 = create_data_subsets(head_df)

df1.to_csv('df_1.csv', index=False)
df2.to_csv('df_2.csv', index=False)
df3.to_csv('df_3.csv', index=False)
df4.to_csv('df_4.csv', index=False)
df5.to_csv('df_5.csv', index=False)

print('Data Saved')
df_list = [df1, df2, df3, df4, df5]


print('GC Cleaned: ', gc.collect())


def train_test_split_df(df):
    print('Creating Train Test Split')
    y = df.Listing_Type.astype('int')
    X = df.drop('Listing_Type', 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.values)
    print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


print('Begin Training:\n\n')
for idx, df in enumerate(df_list):
    print('Set {} Training'.format(idx))
    X_train, X_test, y_train, y_test = train_test_split_df(df)

    # svclassifier = SVC(C=0.1 * (idx + 1),
    #                    random_state=(idx + 1) * 24, probability=True)
    print('Naive Bayes Fitting')
    nb_model.fit(X_train, y_train)
    from sklearn.externals import joblib

    print('Done')
    filename = 'df_nb' + str(idx) + '.pfl'
    joblib.dump(nb_model, filename)
    print('Prediction in progress')
    y_pred = nb_model.predict(X_test)

    print('Confusion Matrix:\n\n')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    import gc

    print('GC Cleaned: ', gc.collect())

end = time.time() - start
print('Total Time: ', end)
