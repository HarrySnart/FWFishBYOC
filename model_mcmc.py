print('beginning job')

# load dependencies

import ads
import pandas as pd
import numpy as np
import bambi as bmb
import os
from ads.dataset.dataset import ADSDataset
from ads.dataset.factory import DatasetFactory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ads.common.auth import default_signer
from sklearn.metrics import root_mean_squared_error


print('packages loaded')

# pull data from object storage

files = ['data/River_Dart_FW_Counts_2015_2025.csv',
'data/River_Exe_FW_Counts_2015_2025.csv',
'data/River_Plym_FW_Counts_2015_2025.csv',
'data/River_Teign_FW_Counts_2015_2025.csv']

ads.set_auth(auth='resource_principal')
bucket_name = '<your bucket here>'
file_name = files[0]
namespace = '<your namespace here>'
data = pd.read_csv(f"oci://{bucket_name}@{namespace}/{file_name}", storage_options=default_signer())

for i in [1,2,3]:
    file_name = files[i]
    dat = pd.read_csv(f"oci://{bucket_name}@{namespace}/{file_name}", storage_options=default_signer())
    data = pd.concat([data,dat])

print('data loaded from object storage')

# do data pre-processing
data = data[['AREA', 'SITE_ID', 'SITE_NAME', 'SURVEY_ID', 'EVENT_DATE',
       'EVENT_DATE_YEAR', 'SAMPLE_CODE', 'SURVEY_RANKED_NGR',
       'SURVEY_RANKED_EASTING', 'SURVEY_RANKED_NORTHING', 'SURVEY_LENGTH',
       'SURVEY_WIDTH', 'SURVEY_AREA', 'FISHED_WIDTH', 'FISHED_AREA',
       'SURVEY_METHOD', 'SURVEY_STRATEGY', 'NO_OF_RUNS', 'SURVEY_SPECIES_ID',
       'SPECIES_ID', 'SPECIES_NAME', 'LATIN_NAME', 'RUN1', 'RUN2', 'RUN3',
       'RUN4', 'RUN5', 'RUN6', 'ALL_RUNS', 'SPCSNO', 'SPCSNO_SE', 'SPCSPV',
       'SPCSPV_SE', 'OBSERVED_ABUNDANCE', 'ZERO_CATCH', 'SURVEY_STATUS',
       'IS_THIRD_PARTY', 'IS_SPECIES_SELECTIVE']]

data=data.reset_index()

fish = data[['AREA', 'SITE_NAME', 'EVENT_DATE',
       'EVENT_DATE_YEAR', 
       'SURVEY_RANKED_EASTING', 'SURVEY_RANKED_NORTHING', 'FISHED_AREA',
       'SURVEY_METHOD', 'SURVEY_STRATEGY', 'NO_OF_RUNS', 'SPECIES_NAME', 'ALL_RUNS']]

fish['Observed'] = fish['ALL_RUNS']/fish['NO_OF_RUNS']
fish['SURVEY_METHOD_STRATEGY'] = fish['SURVEY_METHOD'] + '/'+ fish['SURVEY_STRATEGY']

month = []

for i in range(len(fish)):
    try:
        dt = fish['EVENT_DATE'].loc[i]
        mth = int(dt.split('/')[1])
        month.append(mth)
    except:
        print('issue at index:',i)

fish = fish.drop([331])

fish = fish.reset_index()

month = []

for i in range(len(fish)):
    try:
        dt = fish['EVENT_DATE'].loc[i]
        mth = int(dt.split('/')[1])
        month.append(mth)
    except:
        print('issue at index:',i)

fish['EVENT_MONTH'] = month

winter = []
spring = []
summer = []
autumn = []

for i in range(len(fish)):
    wi = pd.Series([12,1,2])
    sp = pd.Series([3,4,5])
    sm = pd.Series([6,7,8])
    au = pd.Series([9,10,11])
    if pd.Series(fish['EVENT_MONTH'].loc[i]).isin(wi).any():
        winter.append(1)
        spring.append(0)
        summer.append(0)
        autumn.append(0)
    if pd.Series(fish['EVENT_MONTH'].loc[i]).isin(sp).any():
        winter.append(0)
        spring.append(1)
        summer.append(0)
        autumn.append(0)
    if pd.Series(fish['EVENT_MONTH'].loc[i]).isin(sm).any():
        winter.append(0)
        spring.append(0)
        summer.append(1)
        autumn.append(0)
    if pd.Series(fish['EVENT_MONTH'].loc[i]).isin(au).any():
        winter.append(0)
        spring.append(0)
        summer.append(0)
        autumn.append(1)

fish['SEASON_WINTER'] = winter
fish['SEASON_SPRING'] = spring
fish['SEASON_SUMMER'] = summer
fish['SEASON_AUTUMN'] = autumn

drop_sp = pd.Series(['Atlantic salmon','Chub','Dace','Flounder','Gudgeon','Perch','Pike','Rainbow trout','Roach','3-spined stickleback','Grayling'])
merge_l = pd.Series(['Lampetra sp.','Lamprey sp.','Lamprey sp. ammocoetes','Brook lamprey ammocoetes'])
merge_e = pd.Series(['European eel','European eels > elvers','European elvers'])

names_clean = []

for i in range(len(fish)):
    name = fish['SPECIES_NAME'].loc[i]
    if pd.Series(fish['SPECIES_NAME'].loc[i]).isin(drop_sp).any():
        name = 'Other'
    if pd.Series(fish['SPECIES_NAME'].loc[i]).isin(merge_l).any():
        name='Lamprey'
    if pd.Series(fish['SPECIES_NAME'].loc[i]).isin(merge_e).any():
        name='European Eel'
    names_clean.append(name)

fish['SPECIES'] = names_clean

fish_model = fish[['AREA','EVENT_DATE_YEAR',
       'SURVEY_RANKED_EASTING', 'SURVEY_RANKED_NORTHING', 'FISHED_AREA', 'SPECIES', 'Observed', 'SURVEY_METHOD_STRATEGY', 'SEASON_SUMMER', 'SEASON_AUTUMN']]

X = fish_model[['AREA','EVENT_DATE_YEAR',
       'SURVEY_RANKED_EASTING', 'SURVEY_RANKED_NORTHING', 'FISHED_AREA', 'SPECIES', 'SURVEY_METHOD_STRATEGY', 'SEASON_SUMMER', 'SEASON_AUTUMN']]
y = fish_model['Observed']

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=X['SPECIES'],test_size=0.2)

def preprocess_data(x_train,x_test,y_train,y_test, categorical_features, numerical_features, target_variable):
    """
    Preprocesses categorical and numerical features using scikit-learn.

    Args:
        data (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of categorical column names.
        numerical_features (list): List of numerical column names.
        target_variable (str): The name of the target variable.

    Returns:
        tuple: A tuple containing the preprocessor and the transformed data.
    """

    # Create transformers - train
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create column transformer - train
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data - train
    X_transformed = preprocessor.fit_transform(x_train)

    # Get the transformed feature names - train
    categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(categorical_names)

    # Create a DataFrame with transformed features - train
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    X_transformed_df[target_variable] = y_train #add target back in.
    train_out = X_transformed_df.copy()
    
    # Create transformers - test
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create column transformer - test
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data - test
    X_transformed = preprocessor.fit_transform(x_test)

    # Get the transformed feature names - test
    categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(categorical_names)

    # Create a DataFrame with transformed features - test
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)
    X_transformed_df[target_variable] = y_test #add target back in.
    test_out = X_transformed_df.copy()

    return train_out,test_out


# 'SEASON_SUMMER', 'SEASON_AUTUMN'
categorical_features = ['AREA', 'SPECIES','SURVEY_METHOD_STRATEGY']
numerical_features = ['EVENT_DATE_YEAR','SURVEY_RANKED_EASTING', 'SURVEY_RANKED_NORTHING', 'FISHED_AREA']
target_variable = 'Observed'

# Preprocess the entire dataset
train, test = preprocess_data(X_train,X_test,y_train,y_test, categorical_features, numerical_features, target_variable)

train['SEASON_SUMMER'] = X_train['SEASON_SUMMER']
train['SEASON_AUTUMN'] = X_train['SEASON_AUTUMN']
test['SEASON_SUMMER'] = X_test['SEASON_SUMMER']
test['SEASON_AUTUMN'] = X_test['SEASON_AUTUMN']


train = train.rename(columns={'EVENT_DATE_YEAR' :'EVENT_DATE_YEAR', 
'SURVEY_RANKED_EASTING' :'SURVEY_RANKED_EASTING', 
'SURVEY_RANKED_NORTHING':'SURVEY_RANKED_NORTHING',
'FISHED_AREA' :'FISHED_AREA' ,
'AREA_Dart' :'AREA_DART' ,
'AREA_Exe' :'AREA_EXE' ,
'AREA_Plym' :'AREA_PLYM', 
'AREA_Teign':'AREA_TEIGN',
'SPECIES_Brown / sea trout' :'SPECIES_SEA_TROUT' ,
'SPECIES_Bullhead' :'SPECIES_BULLHEAD' ,
'SPECIES_European Eel':'SPECIES_EUROPEAN_EEL',
'SPECIES_Lamprey' :'SPECIES_LAMPREY' ,
'SPECIES_Minnow' :'SPECIES_MINNOW' ,
'SPECIES_Other':'SPECIES_OTHER',
'SPECIES_Stone loach':'SPECIES_STONE_LOACH',
'SURVEY_METHOD_STRATEGY_AC ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_AC_ELECTRIC_FISHING_SINGLE_CATCH',
'SURVEY_METHOD_STRATEGY_DC ELECTRIC FISHING/CATCH PUE/T SAMPLE':'SURVEY_METHOD_DC_ELECTRIC_FISHING_TSAMPLE',
'SURVEY_METHOD_STRATEGY_DC ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_DC_ELECTRIC_FISHING_SINGLE_CATCH',
'SURVEY_METHOD_STRATEGY_ELECTRIC FISHING/CATCH DEPLETION SAMPLE':'SURVEY_METHOD_ELECTRIC_FISHING_DEPLETION_SAMPLE',
'SURVEY_METHOD_STRATEGY_ELECTRIC FISHING/CATCH PUE/T SAMPLE':'SURVEY_METHOD_ELECTRIC_FISHING_TSAMPLE',
'SURVEY_METHOD_STRATEGY_ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_ELECTRIC_FISHING_SINGLE_CATCH',
'SURVEY_METHOD_STRATEGY_PDC ELECTRIC FISHING/CATCH PUE/T SAMPLE':'SURVEY_METHOD_PDC_ELECTRIC_FISHING_TSAMPLE',
'SURVEY_METHOD_STRATEGY_PDC ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_PDC_ELECTRIC_FISHING_SINGLE_CATCH',
'Observed' :'OBSERVED' ,
'SEASON_SUMMER':'SEASON_SUMMER',
'SEASON_AUTUMN':'SEASON_AUTUMN'})


test = test.rename(columns={'EVENT_DATE_YEAR' :'EVENT_DATE_YEAR', 
'SURVEY_RANKED_EASTING' :'SURVEY_RANKED_EASTING', 
'SURVEY_RANKED_NORTHING':'SURVEY_RANKED_NORTHING',
'FISHED_AREA' :'FISHED_AREA' ,
'AREA_Dart' :'AREA_DART' ,
'AREA_Exe' :'AREA_EXE' ,
'AREA_Plym' :'AREA_PLYM', 
'AREA_Teign':'AREA_TEIGN',
'SPECIES_Brown / sea trout' :'SPECIES_SEA_TROUT' ,
'SPECIES_Bullhead' :'SPECIES_BULLHEAD' ,
'SPECIES_European Eel':'SPECIES_EUROPEAN_EEL',
'SPECIES_Lamprey' :'SPECIES_LAMPREY' ,
'SPECIES_Minnow' :'SPECIES_MINNOW' ,
'SPECIES_Other':'SPECIES_OTHER',
'SPECIES_Stone loach':'SPECIES_STONE_LOACH',
'SURVEY_METHOD_STRATEGY_AC ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_AC_ELECTRIC_FISHING_SINGLE_CATCH',
'SURVEY_METHOD_STRATEGY_DC ELECTRIC FISHING/CATCH PUE/T SAMPLE':'SURVEY_METHOD_DC_ELECTRIC_FISHING_TSAMPLE',
'SURVEY_METHOD_STRATEGY_DC ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_DC_ELECTRIC_FISHING_SINGLE_CATCH',
'SURVEY_METHOD_STRATEGY_ELECTRIC FISHING/CATCH DEPLETION SAMPLE':'SURVEY_METHOD_ELECTRIC_FISHING_DEPLETION_SAMPLE',
'SURVEY_METHOD_STRATEGY_ELECTRIC FISHING/CATCH PUE/T SAMPLE':'SURVEY_METHOD_ELECTRIC_FISHING_TSAMPLE',
'SURVEY_METHOD_STRATEGY_ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_ELECTRIC_FISHING_SINGLE_CATCH',
'SURVEY_METHOD_STRATEGY_PDC ELECTRIC FISHING/CATCH PUE/T SAMPLE':'SURVEY_METHOD_PDC_ELECTRIC_FISHING_TSAMPLE',
'SURVEY_METHOD_STRATEGY_PDC ELECTRIC FISHING/SINGLE CATCH SAMPLE':'SURVEY_METHOD_PDC_ELECTRIC_FISHING_SINGLE_CATCH',
'Observed' :'OBSERVED' ,
'SEASON_SUMMER':'SEASON_SUMMER',
'SEASON_AUTUMN':'SEASON_AUTUMN'})

train=train.fillna(0)
test=test.fillna(0)

train['LOG_OBSERVED'] = np.log(train['OBSERVED']+1)
test['LOG_OBSERVED'] = np.log(test['OBSERVED']+1)

print('data prepared')
# build mcmc model

model = bmb.Model("LOG_OBSERVED ~ SURVEY_RANKED_EASTING + SURVEY_RANKED_NORTHING + FISHED_AREA + AREA_DART + AREA_EXE  + AREA_PLYM  + AREA_TEIGN + SPECIES_SEA_TROUT + SPECIES_BULLHEAD  + SPECIES_EUROPEAN_EEL + SPECIES_LAMPREY + SPECIES_MINNOW + SPECIES_OTHER + SPECIES_STONE_LOACH + SEASON_SUMMER + SEASON_AUTUMN",train,  family = "zero_inflated_poisson") #use transformed feature names

results = model.fit(draws=1000,chains=4)

print('mcmc model trained')

test_preds = model.predict(idata=results,data=test,inplace=False,kind='response')
preds = test_preds['posterior_predictive'].to_dataframe().reset_index()
preds['OBSERVED'] = np.exp(preds['LOG_OBSERVED']-1)
mean_preds = np.floor(preds.groupby('__obs__')['OBSERVED'].mean())
test['PREDS'] = mean_preds


print('model validation complete')
print('RMSE: ',root_mean_squared_error(test['PREDS'],test['OBSERVED']))

# push csv data back to object storage
print('writing results back to object storage')

file_name = 'results/posterior_predictive.csv'
preds.to_csv(f"oci://{bucket_name}@{namespace}/{file_name}", index=False, storage_options=default_signer())


print('job complete')
