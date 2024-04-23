"""
Summary:
Reshape play-by-play data to make suitable for RNN model input. Construct some additional features. Train RNN model and use it to predict the 
probability of having a valuable zone entry after every event in each dzone/neutral zone possession.
"""

%reset -sf
import pandas as pd
import numpy as np
import re
import os
import copy
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Masking, SimpleRNN, TimeDistributed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import backend as K
import random

full_pre_entry_build = pd.read_csv(os.path.join(os.getcwd(), 'datasets/playbyplay_data_pre_zone_entry.csv'))

# Define outcome variable and construct features for model ----------------------------------------------------

# make numpy array with the outcome variable, whether a possession led to a valuable zone entry
outcome_var_clean_entry = full_pre_entry_build.groupby(['unique_poss_id'], 
        as_index = False)['clean_success_entry_poss_level'].max()
np_outcome_var_clean_entry = outcome_var_clean_entry['clean_success_entry_poss_level'].to_numpy()


#make flags for different zones of the ice, splitting dzone/neutral zone into 12 zones in a grid
for_model = copy.copy(full_pre_entry_build[['unique_poss_id', 'grp_order', 'Event','X_Coordinate', 
                                  'Y_Coordinate', 'X_Coordinate_2', 'Y_Coordinate_2',
                                  'vert_move', 'lat_move', 'distance', 'active_team_sich']])

for_model['zone_event_end'] = np.select(
    # dzone bottom
    [(for_model['X_Coordinate_2'] <= 37) & (for_model['Y_Coordinate_2'] <= 28),
     (for_model['X_Coordinate_2'] <= 37) & (for_model['Y_Coordinate_2'] > 28) & \
         (for_model['Y_Coordinate_2'] <= 57),
    (for_model['X_Coordinate_2'] <= 37) & (for_model['Y_Coordinate_2'] > 57),
    # dzone top
    (for_model['X_Coordinate_2'] > 37) & (for_model['X_Coordinate_2'] <= 75) & \
        (for_model['Y_Coordinate_2'] <= 28),
     (for_model['X_Coordinate_2'] > 37) & (for_model['X_Coordinate_2'] <= 75) & \
         (for_model['Y_Coordinate_2'] > 28) & (for_model['Y_Coordinate_2'] <= 57),
    (for_model['X_Coordinate_2'] > 37) & (for_model['X_Coordinate_2'] <= 75) & \
        (for_model['Y_Coordinate_2'] > 57),
    # neutral zone bottom
    (for_model['X_Coordinate_2'] > 75) & (for_model['X_Coordinate_2'] <= 100) & \
        (for_model['Y_Coordinate_2'] <= 28),
     (for_model['X_Coordinate_2'] > 75) & (for_model['X_Coordinate_2'] <= 100) & \
         (for_model['Y_Coordinate_2'] > 28) & (for_model['Y_Coordinate_2'] <= 57),
    (for_model['X_Coordinate_2'] > 75) & (for_model['X_Coordinate_2'] <= 100) & \
        (for_model['Y_Coordinate_2'] > 57),
    # neutral zone top
    (for_model['X_Coordinate_2'] > 100) & (for_model['X_Coordinate_2'] <= 125) & \
        (for_model['Y_Coordinate_2'] <= 28),
     (for_model['X_Coordinate_2'] > 100) & (for_model['X_Coordinate_2'] <= 125) & \
         (for_model['Y_Coordinate_2'] > 28) & (for_model['Y_Coordinate_2'] <= 57),
    (for_model['X_Coordinate_2'] > 100) & (for_model['X_Coordinate_2'] <= 125) & \
        (for_model['Y_Coordinate_2'] > 57)
    ]
    ,["dzone_b_left", 'dzone_b_center', 'dzone_b_right',
      "dzone_t_left", 'dzone_t_center', 'dzone_t_right',
      "nzone_b_left", 'nzone_b_center', 'nzone_b_right',
      "nzone_t_left", 'nzone_t_center', 'nzone_t_right'], default=None)

for_model = for_model[['Event', 'zone_event_end', 'vert_move', 'lat_move', 
                       'distance', 'unique_poss_id', 'grp_order', 'active_team_sich']]

# scale all the quantitative predictors to be on the same scale
# use min max scaler instead of standardizing since these variables are all lower bounded by zero and do not follow normal distribution
quant_vars = ['vert_move', 'lat_move', 'distance']
non_quant_vars = [
    'Event', 
    'zone_event_end', 'active_team_sich']
stand = ColumnTransformer(
     [("num", MinMaxScaler(), quant_vars),
     ('pass', 'passthrough',non_quant_vars + ['unique_poss_id', 'grp_order'])])
stand.fit(for_model)
data_stand = pd.DataFrame(stand.transform(for_model), columns = quant_vars + non_quant_vars + \
                          ['unique_poss_id', 'grp_order']).convert_dtypes()

## convert fields to proper type
float64_cols = list(data_stand.select_dtypes(include='Float64'))
data_stand[float64_cols] = data_stand[float64_cols].astype('float64')
string_cols = list(data_stand.select_dtypes(include='string'))
data_stand[string_cols] = data_stand[string_cols].astype('object')
int64_cols = list(data_stand.select_dtypes(include='Int64'))
data_stand[int64_cols] = data_stand[int64_cols].astype('int64')

num_data = copy.copy(data_stand[quant_vars + ['unique_poss_id', 'grp_order']])


# one hot encode zone,event, and situation fields and prepare dataset for modelling
cat_data = copy.copy(data_stand[non_quant_vars])

cat_columns = copy.copy(cat_data.columns + "_var")
og_cat_cols = copy.copy(cat_data.columns)
for col in og_cat_cols:
    cat_data[col] = cat_data[col].astype('category')
    cat_data[col + "_var"] = cat_data[col].cat.codes

enc = OneHotEncoder()

# Passing encoded columns
enc_data = pd.DataFrame(enc.fit_transform(
      cat_data[cat_columns]).toarray())

enc_data.columns = enc.get_feature_names_out().tolist()

final_one_hot_data = pd.concat([num_data,enc_data], axis = 1)

event_values = cat_data['Event'].cat.categories
location_values = cat_data['zone_event_end'].cat.categories
sich_values = cat_data['active_team_sich'].cat.categories


# transform variables and reshape dataset to be inputted into Recurrent Neural Network Model -----------------------------------
# transform long pandas dataframe to 3D numpy array with dimension
# (number of sequences, max sequence length, number of features)
all_cols = list(final_one_hot_data.columns)
remove_cols = ['unique_poss_id', 'grp_order', 'distance']
pred_cols = list(set(all_cols).difference(remove_cols))
pred_cols.sort()
   
np_3d_predictors = [group.values for _, group in final_one_hot_data.\
        groupby('unique_poss_id', as_index = False)[pred_cols]]

# since some sequences are shorter than others, zeropad the numpy array, but instead pad with -999
# do -999 so it doesn't get confused with the 0s from the one hot encoded vars
# need dtype float so it isnt rounded
zeropad_np_3d_predictors = keras.utils.pad_sequences(np_3d_predictors, padding="post", value = -999, dtype='float32')

# make outcome array to match length of input sequences, since i am running many to many RNN
outcomes_m2m = [group.values for _, group in full_pre_entry_build[['unique_poss_id','clean_success_entry_poss_level']].\
        groupby('unique_poss_id', as_index = False)[['clean_success_entry_poss_level']]]
        
zeropad_outcomes_m2m = keras.utils.pad_sequences(outcomes_m2m, padding="post", value = -999, dtype='float32')

# train test split
x_train,x_test, y_train, y_test = train_test_split(zeropad_np_3d_predictors,zeropad_outcomes_m2m, 
                                                   test_size = 0.3, train_size = 0.7, random_state = 1000)\


# code to make neural network reproducible ---------------------------------------------------------
#https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
seed_value= 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

# additionally need the below for reproducability
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
keras.utils.set_random_seed(1337)
tf.config.experimental.enable_op_determinism
# using above code i can get reproducable results for my keras rnn

# run many to many RNN -------------------------------------------------------------------------------------------
# outcome to predict is whether a possession led to a valuable zone entry
# the sequence of all events leading up to a given event are used to predict, after that event happens, whether there will be a valuable zone entry 
# in that possession
# the following information about each prior event is used: location, situation, distance, type of event

# parameter tuning was conducted and concluded that an RNN with one hidden layer with 4 neurons should be used.
# small batch size and many epochs are used to prevent overfitting.
# further parameter tuning should be done to further improve model
model = Sequential()
model.add(Masking(mask_value=-999, input_shape=x_train.shape[1:3]))
model.add(SimpleRNN(4, activation='relu', input_shape=x_train.shape[1:3], return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, epochs=50 ,verbose=2, batch_size = 10)


# evaluate model based on its performance at predicting whether a valuable zone entry will happen as of the last event of a sequence
predictions = model.predict(x_test)
pred_2d = predictions.reshape(predictions.shape[0], predictions.shape[1])
# the last column will be the final predicted prob for each sequence so take that
pred_2d_final_pred = pred_2d[:,-1]

# get the true outcome of each sequence
y_test_2d = y_test.reshape(y_test.shape[0], y_test.shape[1])
y_test_seq_outcome = y_test_2d[:,0]

# calculate AUC score
fpr, tpr, thresh = roc_curve(y_test_seq_outcome,pred_2d_final_pred)
auc_full = auc(fpr,tpr)
print(auc_full)


all_ids_long = final_one_hot_data[['unique_poss_id', 'grp_order']].to_numpy()
poss_ids_temp = final_one_hot_data.\
        drop_duplicates(subset=['unique_poss_id'], keep='last').\
        sort_values(by=['unique_poss_id'])
poss_ids = copy.deepcopy(poss_ids_temp[['unique_poss_id']]).to_numpy()

# use the trained model to make predictions for entire dataset, all sequences
all_predictions = model.predict(zeropad_np_3d_predictors)
all_predictions_np = all_predictions.reshape(all_predictions.shape[0], all_predictions.shape[1])
all_predictions_pd = pd.DataFrame(np.hstack((poss_ids,
    all_predictions.reshape(all_predictions.shape[0], all_predictions.shape[1]))))

# get the prediction of probability of having a valuable zone entry after each event in every possession
new_cols_pd = ["A" + str(x) for x in list(all_predictions_pd.columns)]

all_predictions_pd.columns = new_cols_pd

all_predictions_pd = all_predictions_pd.\
    rename(columns = {'A0':"unique_poss_id"})
all_predictions_long = pd.wide_to_long(all_predictions_pd, ["A"], i="unique_poss_id", j="seq_num").\
    sort_values(by=['unique_poss_id', 'seq_num']).reset_index()
all_predictions_long['lagged_values'] = all_predictions_long.\
    groupby(['unique_poss_id'], as_index=False)['A'].shift(1)

all_predictions_long_filt = all_predictions_long[all_predictions_long.\
    apply(lambda x: x['A'] != x['lagged_values'], axis=1)]
    
all_predictions_long_filt_np = copy.copy(all_predictions_long_filt[['A']]).to_numpy()

all_preds_with_index = pd.DataFrame(
    np.hstack((all_ids_long, all_predictions_long_filt_np))).\
    rename(columns = {0:"unique_poss_id",
                      1:'grp_order',
                      2:'prob_success_entry'}).\
        sort_values(by=['unique_poss_id', 'grp_order'])

# merge predictions with the play by play data containing all events in the defensive and neutral zone
pre_entry_merge = full_pre_entry_build.\
    merge(all_preds_with_index, 
    on = ['unique_poss_id', 'grp_order'],
      how = 'left').\
    merge(copy.copy(for_model[['unique_poss_id', 'grp_order', 'zone_event_end']]), 
    on = ['unique_poss_id', 'grp_order'],
      how = 'left')
    
pre_entry_with_probs = copy.copy(pre_entry_merge[['unique_poss_id', 
    'grp_order','Team', 'Player', 'Event','X_Coordinate', 'Y_Coordinate', 'X_Coordinate_2', 'Y_Coordinate_2',
     'zone_event_end','vert_move', 'lat_move', 'distance', 'prob_success_entry', 'active_team_sich',
     'clean_success_entry_poss_level']])
pre_entry_with_probs['zone_event_start'] = pre_entry_with_probs.groupby(['unique_poss_id'], 
    as_index=False)['zone_event_end'].shift(1)

pre_entry_with_probs['lag_prob'] = pre_entry_with_probs.groupby(['unique_poss_id'], 
    as_index=False)['prob_success_entry'].shift(1)

# calculate how much each event increases the probability of having a valuable zone entry
pre_entry_with_probs = pre_entry_with_probs.\
    assign(prob_change = lambda x: x.prob_success_entry - x.lag_prob)

pre_entry_with_probs.to_csv(os.path.join(os.getcwd(), 'datasets/playbyplay_data_with_zone_entry_probabilities.csv'))
