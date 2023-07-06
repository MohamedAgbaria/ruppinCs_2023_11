#Import necessary libraries
import shared
from pandas import read_csv
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

#################################################################################
# DATA READING & Preprocessing

data = read_csv('logins_kodus_kpi_lrn_left_cellularsales_used_20_12_jan_v2.csv')
shared.Know_Data(data)
data = shared.Data_preprocessing(data)
#################################################################################
# REMOVE BACK OFFICE (KPI = 0)
data = shared.remove_back_office(data, '7')
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
group_divition = shared.create_sub_group(data_trimmed_dropped)
data_trimmed_dropped['sub_group'] = group_divition['sub_group']
data_trimmed_dropped['group'] = group_divition['group']
print(data_trimmed_dropped)
final_df = shared.feature_selection_And_prepare(data_trimmed_dropped, '7')

final_df = shared.data_scaling(final_df)
#final_df.to_csv('data.csv',index=False)

print(final_df)
##################################################################################
#Split the features and target variable
X = final_df.drop('target', axis=1)
y = final_df['target']
################################################################################
# Run the models
shared.Logistic_regression(X, y,'normal')
shared.random_Forest(X, y,'normal')
#shared.nn(X,y)
shared.boosting(X,y)
#final_df.to_csv('dataBeforeTrend.csv',index=False)

#################################################################################
# Trend
final_df = shared.trend(final_df)
X = final_df.drop('target', axis=1)
y = final_df['target']
print("after Trend:")
shared.Logistic_regression(X, y,'trend')
print("after Trend:")
shared.random_Forest(X, y,'trend')
#final_df.to_csv('data.csv',index=False)
