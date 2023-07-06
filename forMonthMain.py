#Import necessary libraries
import pickle
import shared
from pandas import read_csv

#################################################################################
# DATA READING & Preprocessing

data = read_csv('logins_kodus_kpi_lrn_left_cellularsales_used_jan_4m.csv')
shared.Know_Data(data)
data = shared.Data_preprocessing(data)
#################################################################################
# REMOVE BACK OFFICE (KPI = 0)
data = shared.remove_back_office(data,'4')
#################################################################################
# OUTLIERS DETECTION AND HANDLING
print(data.shape)
data_trimmed_dropped, data_trimmed_with_mean = shared.find_skewed_bounaries(data, 6)
shared.outliers_detection(data)
shared.outliers_detection(data_trimmed_with_mean)
shared.outliers_detection(data_trimmed_dropped)
print(data_trimmed_dropped.shape)
#################################################################################

# feature_selection_And_prepare&&Scaling
final_df = shared.feature_selection_And_prepare(data_trimmed_dropped,'4')
final_df = shared.data_scaling(final_df)
print(final_df)
##################################################################################
#Split the features and target variable
X = final_df.drop('target', axis=1)
y = final_df['target']
final_df.to_csv('dataforMonth.csv',index=False)

# Prepare the data for Logistic Regression
X_train_log, X_test_log, Y_train_log, y_test_log = shared.smote_prepare_and_split(X, y)

# Prepare the data for Random Forest
X_trainRANF, X_testRANF, y_trainRANF, y_testRANF = shared.smote_prepare_and_split(X, y)

# Load the trained Logistic Regression model from the pickle file
load_model_for_logestic = pickle.load(open('Logistic_regression_normal.pkl','rb'))

# Load the trained Random Forest model from the pickle file
load_model_for_rf = pickle.load(open('random_Forest_normal.pkl','rb'))

# Use the loaded Logistic Regression model to make predictions
y_pred_prob_log = load_model_for_logestic.predict_proba(X_test_log)
y_pred_log = load_model_for_logestic.predict(X_test_log)

# Use the loaded Random Forest model to make predictions
y_pred_prob_rf= load_model_for_rf.predict_proba(X_testRANF)
y_pred_rf = load_model_for_rf.predict(X_testRANF)

#evaluate the models
shared.AUC_ROC_evaluate(y_pred_prob_log,y_pred_log,y_test_log)
shared.AUC_ROC_evaluate(y_pred_prob_rf,y_pred_rf,y_testRANF)








