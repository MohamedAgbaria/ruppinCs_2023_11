#Import necessary libraries
from collections import Counter
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from imblearn import metrics
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, binarize
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import shap
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam

######################################################################



#Prints information about the provided DataFrame, it will display the first few rows, DataFrame information, and summary statistics.
def Know_Data(df):
    print('know The Data:\n')
    print('The DataFrame:\n', df.head(), '\n', ' DataFrame Info:\n',
          df.info(), '\n', ' DataFrame Describe:\n', df.describe(), '\n')

#The function performs several data preprocessing steps,It checks for duplicates, missing values, replaces NaNs with zeros, performs naive feature selection by dropping unnecessary features, and identifies zero variance features.
def Data_preprocessing(df):
    print('Data_preprocessing:\n')
    # are there any duplicates?,Assuming that all cases in the data set are distinct we are not suppose to have identical or repeating rows. If we locate them we should consider removing them from the DF because they might result in biased model
    print('The Number Of Duplicated Rows :\n', df.duplicated().sum())

    # Check for missing values
    missing = df.isnull()
    print('Number Of Missing Values :\n', missing.sum())
    print('The Missing Values :\n', df[missing.any(axis=1)])
    # Replace Nans with values
    df = replace_na_with_zero(df)

    # Naive Feature Selection,remove unnecessary feature - is there is a feature that doesn't contribute any relevant data to the model we'll simply remove it.
    df.drop('touserid', axis=1, inplace=True)
    df.drop('fromuserid', axis=1, inplace=True)
    df.drop('userid_0', axis=1, inplace=True)
    df.drop('userid_1', axis=1, inplace=True)
    df.drop('userid_2', axis=1, inplace=True)

    # Zero variance features,are feature with a constant value and therefore have no impact on the model predectability.
    print('Number Of Unique Values In Each Column  :\n', df.nunique())

    # return df
    return df

#insert a value instead of a Nan by using the fillna function.
def replace_na_with_zero(df):
    df.fillna(value=0, inplace=True)
    return df

# Outliers Detection using the IQR method viewing in a boxplot
def outliers_detection(df):
    # Drop the 'userid' column from the DataFrame
    df.drop('userid',axis=1,inplace=True)

    # Generate color codes for box plot markers
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(df.columns))]

    # Create a new Figure object
    fig = go.Figure()

    # Add box plot traces for each column in the DataFrame
    fig.add_traces(data=[go.Box(
        y=df.iloc[:, i],
        marker_color=c[i],
        name=df.columns[i])
        for i in range(len(df.columns))
    ])
    # Update the layout of the figure
    fig.update_layout(
        title='Outliers Detection BoxPlot',
    )
    # Show the figure
    fig.show()

#Finds the skewed boundaries for each variable in the provided DataFrame.
def find_skewed_bounaries(df, distance):
    # Create copies of the DataFrame to store the trimmed data
    data_trimmed_dropped = df.copy()
    data_trimmed_with_max = df.copy()
    # Iterate over each variable in the DataFrame
    for variable in df.columns:
        # Calculate the interquartile range (IQR) for the variable
        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
        # Check if the IQR is greater than 3 (indicating skewness)
        if(IQR>3):
            # Calculate the lower and upper boundaries based on the distance multiplier
            lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
            upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

            # Identify outliers using boolean masks
            outliers_for_dropped = np.where(data_trimmed_dropped[variable] > upper_boundary, True,
                                            np.where(data_trimmed_dropped[variable] < lower_boundary, True, False))
            outliers_for_max = np.where(data_trimmed_with_max[variable] > upper_boundary, True,
                                         np.where(data_trimmed_with_max[variable] < lower_boundary, True, False))

            # Remove outliers by filtering the DataFrame
            data_trimmed_dropped = data_trimmed_dropped.loc[~(outliers_for_dropped)]
            # Replace outliers with the variable max
            variable_max = df[variable].max()
            data_trimmed_with_max[variable].loc[outliers_for_max] = variable_max

    # Return the trimmed DataFrames
    return data_trimmed_dropped, data_trimmed_with_max

#REMOVE BACK OFFICE (KPI = 0)
def remove_back_office(df,NumOfMonths):
    # Check the number of months(or the check in the main (7 month) or the for month data)
    if(NumOfMonths=='7'):
        # Filter the DataFrame for any month with non-zero KPI values
        df = df[
            (df['kpi_m_1'] > 0) | (df['kpi_m_2'] > 0) | (df['kpi_m_3'] > 0) | (df['kpi_m_4'] > 0) | (
                        df['kpi_m_5'] > 0) | (
                    df['kpi_m_6'] > 0) | (df['kpi_m_7'] > 0)]
        # Reset the index of the filtered DataFrame
        df.reset_index(inplace=True, drop=True)
    else:
        # Filter the DataFrame for any month with non-zero KPI values (limited to 4 months)
        df = df[
            (df['kpi_m_1'] > 0) | (df['kpi_m_2'] > 0) | (df['kpi_m_3'] > 0) | (df['kpi_m_4'] > 0)]
        # Reset the index of the filtered DataFrame
        df.reset_index(inplace=True, drop=True)

    # Return the DataFrame with back-office data removed
    return df

# Prepares the DataFrame for subgroup analysis by filling missing values with 0 and converting non-zero values to 1.
def df_prepare_for_sub_group(df_copy):
    # Fill NaN values with 0
    df_copy.fillna(0, inplace=True)

    # Convert non-zero values to 1
    for row in range(0, len(df_copy)):
        for column in df_copy.columns:
            if df_copy[column].iloc[row] != 0:
                df_copy[column].iloc[row] = 1

    # Return the prepared DataFrame
    return df_copy


def create_sub_group(df):
    # Select the columns of interest from the DataFrame
    df_copy = df[['logins_m_1', 'logins_m_2', 'logins_m_3', 'logins_m_4', 'logins_m_5', 'logins_m_6', 'logins_m_7']]

    # Create 'sub_group' and 'group' columns and initialize them with 0
    df_copy['sub_group'] = 0
    df_copy['group'] = 0

    ####################################################3
    # Prepare the DataFrame for subgroup analysis
    df_copy = df_prepare_for_sub_group(df_copy)
    ####################################################3
    # Assign power values to the non-zero elements in the DataFrame
    power = 0
    for column in df_copy.columns:
        df_copy[column] = df_copy[column].replace([1], pow(2, power))
        power += 1

    # Determine the subgroups based on the sum of values in each row
    for row in range(0, len(df_copy)):
        result = 0
        for column in df_copy.columns:
            result += df_copy[column].iloc[row]
        if (result == 127) | (result == 126) | (result == 124) | (result == 120):
            df_copy['group'].iloc[row] = 'Stayed'
            df_copy['sub_group'].iloc[row] = '1'
        if (result == 112) or (result == 1) or (result == 3) or (result == 64) or (result == 96):
            df_copy['group'].iloc[row] = 'Noise'
            df_copy['sub_group'].iloc[row] = '2'

        if (result == 6) or (result == 12) or (result == 24) or (result == 48):
            df_copy['group'].iloc[row] = 'Left to drop'
            df_copy['sub_group'].iloc[row] = '3'
        if (result == 7) or (result == 14) or (result == 28) or (result == 56) or (result == 15) or (result == 30) or (
                result == 60) or (result == 31) or (result == 62) or (result == 63):
            df_copy['group'].iloc[row] = 'Left'
            if (result == 7):
                df_copy['sub_group'].iloc[row] = '4'
            if (result == 14):
                df_copy['sub_group'].iloc[row] = '5'
            if (result == 28):
                df_copy['sub_group'].iloc[row] = '6'
            if (result == 56):
                df_copy['sub_group'].iloc[row] = '7'
            if (result == 15):
                df_copy['sub_group'].iloc[row] = '8'
            if (result == 30):
                df_copy['sub_group'].iloc[row] = '9'
            if (result == 60):
                df_copy['sub_group'].iloc[row] = '10'
            if (result == 31):
                df_copy['sub_group'].iloc[row] = '11'
            if (result == 62):
                df_copy['sub_group'].iloc[row] = '12'
            if (result == 63):
                df_copy['sub_group'].iloc[row] = '13'


    # Print the resulting DataFrame
    print(df_copy)
    # Return the DataFrame with 'sub_group' and 'group' columns
    return df_copy[['sub_group','group']]

#Returns a list of feature names that correspond to subgroups 5, 8, 9, 11, 12, and 13.
def feature_to_sub_group_5_8_9_11_12_13():
   return ['logins_m_2', 'logins_m_3', 'logins_m_4', 'kudos_t_m_2', 'kudos_t_m_3', 'kudos_t_m_4', 'kudos_f_m_2', 'kudos_f_m_3', 'kudos_f_m_4', 'kpi_m_2', 'kpi_m_3', 'kpi_m_4', 'lrn_m_2', 'lrn_m_3', 'lrn_m_4', 'lrn_t_m_2', 'lrn_t_m_3', 'lrn_t_m_4']

#Returns a list of feature names that correspond to subgroups 6,10.
def feature_to_sub_group_6_10():
    return ['logins_m_3', 'logins_m_4', 'logins_m_5', 'kudos_t_m_3', 'kudos_t_m_4', 'kudos_t_m_5', 'kudos_f_m_3', 'kudos_f_m_4', 'kudos_f_m_5', 'kpi_m_3', 'kpi_m_4', 'kpi_m_5', 'lrn_m_3', 'lrn_m_4', 'lrn_m_5', 'lrn_t_m_3', 'lrn_t_m_4', 'lrn_t_m_5']

#Returns a list of feature names that correspond to subgroups 1,7.
def feature_to_sub_group_1_7():
    return ['logins_m_4', 'logins_m_5', 'logins_m_6', 'kudos_t_m_4', 'kudos_t_m_5', 'kudos_t_m_6', 'kudos_f_m_4',
         'kudos_f_m_5', 'kudos_f_m_6', 'kpi_m_4', 'kpi_m_5', 'kpi_m_6', 'lrn_m_4', 'lrn_m_5', 'lrn_m_6', 'lrn_t_m_4',
         'lrn_t_m_5', 'lrn_t_m_6']

def feature_selection_And_prepare(df,NumOfMonths):
    # If NumOfMonths is '4', perform feature selection for NumOfMonths = 4
    if(NumOfMonths=='4'):
        # Update values in the 'logins_m_4' column
        for row in range(0, len(df)):
                if df['logins_m_4'].iloc[row] != 0:
                    df['logins_m_4'].iloc[row] = 0
                else:
                    df['logins_m_4'].iloc[row] = 1

        # Create the final DataFrame with selected features and target variable
        final_df = pd.DataFrame()
        final_df[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df.drop(['logins_m_4','kudos_t_m_4','kudos_f_m_4','kpi_m_4','lrn_m_4','lrn_t_m_4'],axis=1)
        final_df['target']=df['logins_m_4']
        final_df.reset_index(inplace = True,drop=True)
        print(final_df)

    # For other values of NumOfMonths
    else:
        # Feature selection for sub_groups 1 and 7
        columns_names_for_sub_1_7 = feature_to_sub_group_1_7()

        # Process sub_group 1
        X_1 = pd.DataFrame()
        df_sub_1 = df[df['sub_group']=='1']
        df_sub_1.reset_index(inplace=True)
        df_sub_1 = df_sub_1.loc[:,columns_names_for_sub_1_7]
        X_1[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_1
        X_1['target'] = 0

        # Process sub_group 7
        X_7 = pd.DataFrame()
        df_sub_7 = df[df['sub_group'] == '7']
        df_sub_7.reset_index(inplace=True)
        df_sub_7 = df_sub_7.loc[:,columns_names_for_sub_1_7]
        X_7[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_7
        X_7['target'] = 1
        ###########################################################################################
        # Process sub_group 7
        X_4 = pd.DataFrame()
        df_sub_4 = df[df['sub_group']=='4']
        df_sub_4.reset_index(inplace=True)
        df_sub_4 = df_sub_4.loc[:,['logins_m_1', 'logins_m_2', 'logins_m_3', 'kudos_t_m_1', 'kudos_t_m_2', 'kudos_t_m_3', 'kudos_f_m_1', 'kudos_f_m_2', 'kudos_f_m_3', 'kpi_m_1', 'kpi_m_2', 'kpi_m_3', 'lrn_m_1', 'lrn_m_2', 'lrn_m_3', 'lrn_t_m_1', 'lrn_t_m_2', 'lrn_t_m_3']]
        X_4[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_4
        X_4['target'] = 1
        ###########################################################################################
        # Feature selection for sub_groups 5, 8, 9, 11, 12, 13
        culomns_names = feature_to_sub_group_5_8_9_11_12_13()
        X_5 = pd.DataFrame()
        X_8 = pd.DataFrame()
        X_9 = pd.DataFrame()
        X_11 = pd.DataFrame()
        X_12 = pd.DataFrame()
        X_13 = pd.DataFrame()

        df_sub_5 = df[df['sub_group'] == '5']
        df_sub_5.reset_index(inplace=True)
        df_sub_5 = df_sub_5.loc[:,culomns_names]
        df_sub_8 = df[df['sub_group'] == '8']
        df_sub_8.reset_index(inplace=True)
        df_sub_8 = df_sub_8.loc[:,culomns_names]
        df_sub_9 = df[df['sub_group'] == '9']
        df_sub_9.reset_index(inplace=True)
        df_sub_9 = df_sub_9.loc[:,culomns_names]
        df_sub_11 = df[df['sub_group'] == '11']
        df_sub_11.reset_index(inplace=True)
        df_sub_11 = df_sub_11.loc[:,culomns_names]
        df_sub_12 = df[df['sub_group'] == '12']
        df_sub_12.reset_index(inplace=True)
        df_sub_12 = df_sub_12.loc[:,culomns_names]
        df_sub_13 = df[df['sub_group'] == '13']
        df_sub_13.reset_index(inplace=True)
        df_sub_13 = df_sub_13.loc[:,culomns_names]
        X_5[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_5
        X_8[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_8
        X_9[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_9
        X_11[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_11
        X_12[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_12
        X_13[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_13
        X_5['target'] = 1
        X_8['target'] = 1
        X_9['target'] = 0
        X_11['target'] = 0
        X_12['target'] = 0
        X_13['target'] = 0

        ###########################################################################################
        # Feature selection for sub_groups 6+10
        columns_names_for_sub_6_10 = feature_to_sub_group_6_10()
        X_6 = pd.DataFrame()
        X_10 = pd.DataFrame()
        df_sub_6 = df[df['sub_group'] == '6']
        df_sub_6.reset_index(inplace=True)
        df_sub_6 = df_sub_6.loc[:,columns_names_for_sub_6_10]
        df_sub_10 = df[df['sub_group'] == '10']
        df_sub_10.reset_index(inplace=True)
        df_sub_10 = df_sub_10.loc[:,columns_names_for_sub_6_10]
        X_6[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_6
        X_10[['final_logins_m_1', 'final_logins_m_2', 'final_logins_m_3', 'final_kudos_t_m_1', 'final_kudos_t_m_2',
             'final_kudos_t_m_3', 'final_kudos_f_m_1', 'final_kudos_f_m_2', 'final_kudos_f_m_3', 'final_kpi_m_1',
             'final_kpi_m_2', 'final_kpi_m_3', 'final_lrn_m_1', 'final_lrn_m_2', 'final_lrn_m_3', 'final_lrn_t_m_1',
             'final_lrn_t_m_2', 'final_lrn_t_m_3']] = df_sub_10
        X_6['target'] = 1
        X_10['target'] = 0

        ###########################################################################################
        ###########################################################################################
        # Concatenate all processed sub-group DataFrames
        final_df = pd.concat([X_1,X_4,X_5,X_6,X_7,X_8,X_9,X_10,X_11,X_12,X_13])
        final_df.reset_index(inplace = True,drop=True)
    return final_df


# Scaling - MinMax, This technique re-scales a feature or observation value with distribution value between 0 and 1. :
def data_scaling(df):
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
    return df




#Prepares and splits the data & use SMOTE oversampling technique(oversmaple the minory class) to handle the Imbalanced data
def smote_prepare_and_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    smote = SMOTE(random_state=1)
    counter = Counter(y_train)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, y_train)
    counter = Counter(Y_train_resampled)
    return X_train_resampled, X_test, Y_train_resampled, y_test

# run  Logistic regression model,evaluates the results usuing AUC_ROC_evaluate function,shap,pickle
def Logistic_regression(X,y,status):
    # Split the data into train and test sets
    X_train_logr, X_test_logr, y_train_logr, y_test_logr = smote_prepare_and_split(X, y)

    # Create a logistic regression model
    logmodel = LogisticRegression()

    # Fit the model on the training data
    logmodel.fit(X_train_logr, y_train_logr)

    # Save the trained model using pickle
    pickle.dump(logmodel, open(f'Logistic_regression_{status}.pkl', 'wb'))

    # Predict probabilities and class labels on the test set
    probability_ = logmodel.predict_proba(X_test_logr)
    log_pred = logmodel.predict(X_test_logr)

    # Evaluate the model's performance
    print('Logistic_regression evaluate results:')
    AUC_ROC_evaluate(probability_,log_pred,y_test_logr)

    # Generate SHAP explanations
    explainer = shap.Explainer(logmodel, X_train_logr)  # Assuming X_train_logr is the training data used for the model
    shap_values = explainer.shap_values(X_test_logr)
    X_test_logr_resetd_index = X_test_logr.reset_index()
    row_index = 487  # Index of the specific row you want to explain
    row_to_explain = X_test_logr.iloc[row_index]  # Assuming X_test_logr is your test data

    # Generate force plot for a specific row
    f = shap.force_plot(explainer.expected_value, shap_values[row_index], row_to_explain)
    shap.save_html("Logistic_regression shap results.html", f)

    # Generate summary plot of SHAP values
    shap.summary_plot(shap_values, X_test_logr)

#Displays the feature importances of a given model.
def show_importance(model, X):
    # Retrieve the feature importances from the model
    importances = model.feature_importances_
    # Create a Series with feature importances and column names
    model_importances = pd.Series(importances, index=X.columns)
    # Sort the importances in descending order
    model_importances = model_importances.sort_values(ascending=False)

    # Print the feature importances
    print('The Feature Importances')
    print(model_importances)

    # Create a bar plot to visualize the feature importances
    fig, ax = plt.subplots()
    model_importances.plot.bar(ax=ax)
    ax.set_title("Feature Names")
    ax.set_ylabel("Feature Importances")
    plt.show()


# run  random Forest model,evaluates the results usuing AUC_ROC_evaluate function,shap,pickle
def random_Forest(X,y,status):
    # Split the data into train and test sets
    X_trainRANF, X_testRANF, y_trainRANF, y_testRANF = smote_prepare_and_split(X, y)

    # Create a random forest classifier
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)

    # Fit the model on the training data
    rforest.fit(X_trainRANF, y_trainRANF)

    # Show feature importances
    show_importance(rforest, X_trainRANF)

    # Save the trained model using pickle
    pickle.dump(rforest, open(f'random_Forest_{status}.pkl', 'wb'))

    # Predict probabilities and class labels on the test set
    rforest_proba = rforest.predict_proba(X_testRANF)
    rforest_pred = rforest.predict(X_testRANF)

    # Evaluate the model's performance
    print('random_Forest evaluate results:')
    AUC_ROC_evaluate(rforest_proba,rforest_pred,y_testRANF)

    # Create an explainer object using SHAP to explain the model's predictions
    explainer = shap.Explainer(rforest, X_testRANF)  # Assuming X_train_logr is the training data used for the model

    # List of row indices to explain using SHAP
    rows_list_to_shap = [487,489,490,550,14,473,480,484]

    # Explain each row and save the force plots as HTML files
    for i in rows_list_to_shap:
      row_to_explain = X_testRANF.iloc[i]  # Assuming X_test is your test data
      shap_values = explainer.shap_values(row_to_explain, check_additivity=False)
      print(shap_values)
      shap.initjs()
      f = shap.force_plot(explainer.expected_value[1], shap_values[1], row_to_explain)
      shap.save_html(f"random_Forest shap results {i}.html", f)

    # Generate summary plot of SHAP values for all features
    shap_values_summary = explainer.shap_values(X_testRANF, check_additivity=False)
    shap.summary_plot(shap_values_summary, X_testRANF)
    plt.show()


#evaluate our models,display Auc,classification_report with diffrent ThreShold
def AUC_ROC_evaluate(probability, prediction, y_test):
    # Create Series for probability, prediction, and actual values
    y_test_prob_0 = pd.Series(probability[:, 0], name='probability_0', index=y_test.index)
    y_test_prob_1 = pd.Series(probability[:, 1], name='probability_1', index=y_test.index)
    y_test_pred = pd.Series(prediction, name='prediction', index=y_test.index)

    # Create a DataFrame to store the evaluation results
    uni_test_results = pd.DataFrame(
        data={'probability_0': y_test_prob_0, 'probability_1': y_test_prob_1, 'prediction': y_test_pred,
              'actual': y_test})

    # Calculate metrics using different threshold values
    y_pred_prob = uni_test_results.probability_1
    y_test_pred = uni_test_results.prediction
    y_pred_03 = binarize(X=[y_pred_prob], threshold=0.3)[0]
    y_pred_03 = pd.Series(y_pred_03)
    y_pred_08 = binarize(X=[y_pred_prob], threshold=0.8)[0]
    y_pred_08 = pd.Series(y_pred_08)

    # Print classification report for different threshold values
    print(classification_report(y_test, y_test_pred))
    print('ThreShold = 0.3')
    print(classification_report(y_test, y_pred_03))
    print('ThreShold = 0.8')
    print(classification_report(y_test, y_pred_08))

    # Calculate and print the AUC-ROC score
    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    print('AUC : ', AUC)

#Computes the trend features based on the given columns in the DataFrame.
def trend(df):
    df['Trend_logins_1']=df['final_logins_m_2']-df['final_logins_m_1']
    df['Trend_logins_2'] = df['final_logins_m_3'] - df['final_logins_m_2']
    df['Trend_kudos_t_m_1'] = df['final_kudos_t_m_2'] - df['final_kudos_t_m_1']
    df['Trend_kudos_t_m_2'] = df['final_kudos_t_m_3'] - df['final_kudos_t_m_2']
    df['Trend_kudos_f_m_1'] = df['final_kudos_f_m_2'] - df['final_kudos_f_m_1']
    df['Trend_kudos_f_m_2'] = df['final_kudos_f_m_3'] - df['final_kudos_f_m_2']
    df['Trend_kpi_m_1'] = df['final_kpi_m_2'] - df['final_kpi_m_1']
    df['Trend_kpi_m_2'] = df['final_kpi_m_3'] - df['final_kpi_m_2']
    df['Trend_lrn_m_1'] = df['final_lrn_m_2'] - df['final_lrn_m_1']
    df['Trend_lrn_m_2'] = df['final_lrn_m_3'] - df['final_lrn_m_2']
    df['Trend_lrn_t_m_1'] = df['final_lrn_t_m_2'] - df['final_lrn_t_m_1']
    df['Trend_lrn_t_m_2'] = df['final_lrn_t_m_3'] - df['final_lrn_t_m_2']

    return df


#Trains a neural network model using the given features (X) and target variable (y)
def nn(X,y):
    # Split the data into train and test sets
    X_trainRANF, X_testRANF, y_trainRANF, y_testRANF = smote_prepare_and_split(X, y)
    # Define the model architecture
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(X_trainRANF.shape[1],)))
    model.add(Dense(64, activation='relu', ))
    #model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))  # Output layer with a single unit


    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Train the model
    model.fit(X_trainRANF, y_trainRANF, batch_size=32, epochs=10, verbose=1)



    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_testRANF, y_testRANF)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)



#Trains an AdaBoost classifier using the given features (X) and target variable (y).
def boosting(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = smote_prepare_and_split(X, y)

    # Create an AdaBoost classifier
    adaBoost = AdaBoostClassifier()

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.01, 0.001]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(adaBoost, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train the model with the best hyperparameters
    adaBoost_best = AdaBoostClassifier(**best_params)
    adaBoost_best.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = adaBoost_best.predict(X_test)
    proba = adaBoost_best.predict_proba(X_test)


    # Evaluate the model on the test set
    AUC_ROC_evaluate(proba,y_pred,y_test)

    return adaBoost_best









