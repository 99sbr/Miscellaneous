import time
import warnings

start = time.time()
warnings.filterwarnings("ignore")  # Ignoring unnecessory warnings
from numerical_data_preprocessing import preprocessing_numerics_df
from categorical_data_preprocessing import preprocess_categorical_df
import pandas as pd

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
#
print("------- LOADING DATA ------")
listings = pd.read_csv('/Users/subir/Codes/Miscellaneous/HaikuJAM/listings_test_send.csv', delimiter=';')

print('Data Shape: ', listings.shape)
guid = listings['GUID']
train_df = listings.drop(['Unnamed: 0', 'GUID'], 1)

columns_to_drop = ['Has_Availability',
                   'Square_Feet',
                   'License',
                   'Host_Acceptance_Rate',
                   'Monthly_Price',
                   'Weekly_Price',
                   'Neighbourhood_Group_Cleansed',
                   'Jurisdiction_Names',
                   'Security_Deposit',
                   'Notes']

for col in columns_to_drop:
    train_df.drop([col], 1, inplace=True)

numerical_cols = [
    'Host_Since', 'Host_Listings_Count', 'Accommodates', 'Cleaning_Fee',
    'Guests_Included', 'Maximum_Nights', 'Calendar_last_Scraped',
    'First_Review', 'Last_Review', 'Geolocation'
]

all_numerical_columns = set(train_df.select_dtypes(exclude=['object']).columns.to_list() + numerical_cols)

numerical_df = train_df[all_numerical_columns]
print('Numerical Data Shape: ', numerical_df.shape)
print('Preprocessing Numerical Data in Progress')
numerical_df = preprocessing_numerics_df(numerical_df)

all_object_columns = list(set(train_df.columns.to_list()) - set(all_numerical_columns))
categorical_df = train_df[all_object_columns]
print('Categorical Data Shape: ', categorical_df.shape)

print('Preprocessing Categorical Data in Progress')
categorical_df = preprocess_categorical_df(categorical_df)

assert categorical_df.shape[0] == numerical_df.shape[0]
head_df = categorical_df.join(numerical_df)
print('Preprocessing Done')
import joblib

# load the model from disk
class_mapping = {1: 'Good', 0: 'Bad'}
models = ['df_nb0.pfl', 'df_nb1.pfl', 'df_nb2.pfl', 'df_nb3.pfl', 'df_nb4.pfl']
probability = []
for model in models:
    loaded_model = joblib.load(model)
    pred = loaded_model.predict(head_df)
    print(pred)
    probability.append(pred)

listing_type = []
for i in range(0, len(guid)):
    preds = [x[i] for x in probability]
    listing_type.append(max(preds, key=preds.count))

result = pd.DataFrame()
result['GUID'] = guid
result['Listing_Type'] = [class_mapping[x] for x in listing_type]
result.to_csv('test_pred.csv', index=False)
