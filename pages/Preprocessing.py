import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64 # Standard Python Module 
from io import StringIO, BytesIO # Standard Python Module
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


# 개발해야 하는 부분 1. outlier 개발 
#                    2. missing value -> groupby


# 파일 다운로드 함수 
def generate_excel_download_link(df):
    # Credit Excel: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5
    towrite = BytesIO()
    df.to_excel(towrite, encoding="utf-8", index=False, header=True)  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_download.xlsx">Download Excel File</a>'
    return st.markdown(href, unsafe_allow_html=True)

def generate_html_download_link(fig):
    # Credit Plotly: https://discuss.streamlit.io/t/download-plotly-plot-as-html/4426/2
    towrite = StringIO()
    fig.write_html(towrite, include_plotlyjs="cdn")
    towrite = BytesIO(towrite.getvalue().encode())
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download Plot</a>'
    return st.markdown(href, unsafe_allow_html=True)


# 함수목록

# 1. 데이터 로드
@st.cache
def load_dataframe(upload_file):
    global df, columns
    try:
        df = pd.read_csv(upload_file)
        
    except Exception as e:
        print(e)
        df = pd.read_excel(upload_file)

    columns = list(df.columns)
    columns.insert(0, None)

    return df, columns 


def target_dataframe(upload_target):
    global target
    try:
        target = pd.read_csv(upload_target)
        
    except Exception as e:
        print(e)
        target = pd.read_excel(upload_target)

    return target 



# drop columns

def drop_df(df):
    global features
    select_dropOrnot = st.selectbox("Drop or Not?",("No", "Yes"))
    if select_dropOrnot == "Yes":
        select_drop_columns = st.multiselect("Select Drop Columns", df.columns)
        if select_drop_columns is not None:

            features = df.drop(axis=1, columns=select_drop_columns)
            st.subheader("Transform Features")
            st.dataframe(features)
            return features
        else:
            features = df
            st.subheader("No change Features")
            st.dataframe(features)
            return features
    else:
        features = df
        st.subheader("No change Features")
        st.dataframe(features)
        return features
    
# Drop_na
def Drop_na(df):
    global features
    drop_columns = st.selectbox("Drop NA Columns?",('Yes', 'No'))
    if drop_columns == "Yes":
        drop_method = st.selectbox("Drop Method",('any', 'all'))
        if drop_method is not None:
            try:
                drop_axis = st.selectbox("Select Axis", (0, 1))
                features = df.dropna(how=drop_method, axis=drop_axis)
                st.success('Complete Drop NA Columns', icon="✅")
                st.subheader("Transform Features")
                st.dataframe(features)
                return features
            except Exception as e:
                print(e)
    else:
        st.subheader("No change Features")
        st.dataframe(features)
        return features


# 2. 타겟, 피쳐 나누기
def split_x_y(df):
    global features, target, df_numeric, df_object, df_datetime
    df_numeric = features.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
    df_object = features.select_dtypes(include = 'object').columns.to_list()
    df_datetime = features.select_dtypes(include = 'datetime').columns.to_list()
    select_target= st.selectbox("Select Target", columns)
    if select_target is not None:
        try:
            features = df.drop([select_target],axis=1)
            target = df[select_target]
            st.success('Complited Split your target', icon="✅")
            st.subheader("Features")
            st.dataframe(features)
            st.subheader("Target")
            st.dataframe(target)
            df_numeric = features.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
            df_object = features.select_dtypes(include = 'object').columns.to_list()
            df_datetime = features.select_dtypes(include = 'datetime').columns.to_list()
            return features, target
        except Exception as e:
            print(e)
        
    else:
        try:
            st.info("Upload Your Target Data", icon='ℹ️')
            target = st.file_uploader("", type=['xlsx', 'xls', 'csv'], accept_multiple_files=False, key='<target>')
            target_dataframe(target)
            if target.shape[0] != features.shape[0] or target.shape[1] != 1:
                try:
                    st.error('Target Size & features Size Error',icon='🚨')
                except Exception as e:
                    print(e)
            else:
                try:
                    st.subheader("Features")
                    st.dataframe(features)
                    st.subheader("Target")
                    st.dataframe(target)
                    df_numeric = features.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                    df_object = features.select_dtypes(include = 'object').columns.to_list()
                    df_datetime = features.select_dtypes(include = 'datetime').columns.to_list()
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
        return features, target, df_numeric, df_object, df_datetime
    

        

# 3. train_test_split
def split_train_test_split(features, target):
    global X_train, X_test, X_val, y_train, y_test, y_val, validation_select, test_size_input, val_size_input
    validation_select = st.selectbox("You want validation Data?",('Yes', 'No'))
    stratify_select = st.selectbox("Using Stratify?", ('No', 'Yes'))
    
    if validation_select == 'Yes':
        if stratify_select == 'No':
            df = pd.concat([features, target], axis=1)
            df_columns = df.columns.to_list()

            test_size_input = st.slider("Pleas Select Test Size", min_value=0.1, max_value=0.9, format='%.2f')
            val_size_input = st.slider("Pleas Select Val Size", min_value=0.1, max_value=0.9, format='%.2f')
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size_input, random_state=42)       
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_input, random_state=42)

            features = pd.concat([X_train, X_test, X_val], axis=0)
            target = pd.concat([y_train, y_test, y_val], axis=0)
            st.info("Check Your Data Split Size", icon="ℹ️")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Size", len(features))
            col2.metric("Train Size", len(X_train))
            col3.metric("Validation Size", len(X_val))
            col4.metric("Test Size", len(X_test))
            return features, X_train, X_test, y_train, y_test, X_val, y_val
        
        else:
            df = pd.concat([features, target], axis=1)
            df_columns = df.columns.to_list()

            stratify_columns = st.multiselect("Select Stratify Column", df_columns)
            stratify_columns_count = len(stratify_columns)
            if stratify_columns_count == 1:

                test_size_input = st.slider("Pleas Select Test Size", min_value=0.1, max_value=0.9, format='%.2f')
                val_size_input = st.slider("Pleas Select Val Size", min_value=0.1, max_value=0.9, format='%.2f')
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size_input, stratify=df[stratify_columns], random_state=42)
                df_train_test = pd.concat([X_train, y_train], axis=1)       
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_input, stratify=df_train_test[stratify_columns],random_state=42)
                # value_counst() 확인한 결과 2번 나눠서 그런지 일부 비율이 안맞는데 맞출 수 있는 방법 고민해봐야함.
                                
                features = pd.concat([X_train, X_test, X_val], axis=0)
                target = pd.concat([y_train, y_test, y_val], axis=0)
                st.info("Check Your Data Split Size", icon="ℹ️")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Size", len(features))
                col2.metric("Train Size", len(X_train))
                col3.metric("Validation Size", len(X_val))
                col4.metric("Test Size", len(X_test))
                return features, X_train, X_test, y_train, y_test, X_val, y_val
           
            elif stratify_columns_count >= 2:
                features['multi_columns_Stratify'] = ""
                for i in stratify_columns:
                    features['multi_columns_Stratify'] = features['multi_columns_Stratify'] + "_" +df[i].astype(str)
                   
                              
                    
                test_size_input = st.slider("Pleas Select Test Size", min_value=0.1, max_value=0.9, format='%.2f')
                val_size_input = st.slider("Pleas Select Val Size", min_value=0.1, max_value=0.9, format='%.2f')
                try:
                    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size_input, stratify=features['multi_columns_Stratify'], random_state=42)
                    df_train_test = pd.concat([X_train, y_train], axis=1)       
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_input, stratify=X_train['multi_columns_Stratify'],random_state=42)
                    # value_counst() 확인한 결과 2번 나눠서 그런지 일부 비율이 안맞는데 맞출 수 있는 방법 고민해봐야함.
                except Exception as e:
                    st.error("The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.")
                                
                features = pd.concat([X_train, X_test, X_val], axis=0)
                target = pd.concat([y_train, y_test, y_val], axis=0)
                st.info("Check Your Data Split Size", icon="ℹ️")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Size", len(features))
                col2.metric("Train Size", len(X_train))
                col3.metric("Validation Size", len(X_val))
                col4.metric("Test Size", len(X_test))
                     
                    
                
                
                
            else:   
                pass
       
    else:
        test_size_input = st.slider("Pleas Select Test Size", min_value=0.1, max_value=0.9, format='%.2f')
        df = pd.concat([features, target], axis=1)
        df_columns = df.columns.to_list()
        df_columns.insert(0, 'None')
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size_input, random_state=42)
        features = pd.concat([X_train, X_test], axis=0)
        target = pd.concat([y_train, y_test], axis=0)
        st.info("Check Your Data Split Size", icon="ℹ️")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Size", len(features))
        col2.metric("Train Size", len(X_train))
        col3.metric("Validation Size", len(X_test))
        return features, X_train, X_test, y_train, y_test
       



# Fill_Na(only numeric)
def fill_na(df):
    global features, X_train, X_test, y_train, y_test, X_val, y_val, fill_columns, validation_select
    df_numeric = features.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
    df_object = features.select_dtypes(include = 'object').columns.to_list()
    df_datetime = features.select_dtypes(include = 'datetime').columns.to_list()
    features_isnull = features.isnull().sum().sum()
    fill_columns = st.selectbox("Fill NA Columns?",('Yes', 'No'))


    if fill_columns == 'No':
        if features_isnull == 0:
            st.success('There is not any NA value in your dataset.', icon="✅")
            st.subheader("Features")
            st.dataframe(features)
            return features
        else:
            st.error("You have to Handling of Null Values", icon="🚨")
            isnull = features.isnull().sum().reset_index()
            isnull = isnull.rename(columns = {'index':'Column', 0:'Number of null values'})
            st.subheader("DATA MISSING VALUES")
            st.dataframe(isnull)
            st.subheader("Only Numeric Columns")
            if features[df_numeric].isnull().sum().sum() == 0:
                st.success('There is not any NA value in your dataset.', icon="✅")
            else:
                st.error("You have to Handling of Null Values", icon="🚨")
            
    else:
        if df[df_numeric].isnull().sum().sum() != 0:
            st.markdown('')
            st.subheader("DATA MISSING VALUES")
            st.dataframe(df.isnull().sum().reset_index().rename(columns = {'index':'Column', 0:'Number of null values'}))
            st.markdown('')
            st.subheader("Fill Only Numeric Columns")
            groupOrNongroup = st.radio("Na Values Search Method", ('Column Values', 'Groupby Values'), horizontal=True)
            if groupOrNongroup == "Column Values":
                fill_na_columns = st.selectbox("How to FillNA",(0, 'mean','min','max','mode','ffill', 'bfill','interpolate'))
                st.markdown('')
                if validation_select == 'Yes':
                    if fill_na_columns == 'mean':
                        st.subheader("Numeric Columns Mean Values")
                        st.dataframe(X_train[df_numeric].mean().reset_index().rename(columns = {'index':'Columns', 0:'Columns Mean Values'}))
                        X_train[df_numeric] = X_train[df_numeric].fillna(X_train[df_numeric].mean())
                        X_test[df_numeric] = X_test[df_numeric].fillna(X_train[df_numeric].mean())
                        X_val[df_numeric] = X_val[df_numeric].fillna(X_train[df_numeric].mean())
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test, X_val], axis=0)
                        st.dataframe(features)

                        return features, X_train, X_test, y_train, y_test, X_val, y_val

                    elif fill_na_columns == 'min':
                        st.subheader("Numeric Columns Min Values")
                        st.dataframe(X_train[df_numeric].min().reset_index().rename(columns = {'index':'Columns', 0:'Columns Min Values'}))
                        X_train[df_numeric] = features[df_numeric].fillna(features[df_numeric].min())
                        X_test[df_numeric] = X_test[df_numeric].fillna(features[df_numeric].min())
                        X_val[df_numeric] = X_val[df_numeric].fillna(features[df_numeric].min())
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test, X_val], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test, X_val, y_val



                    elif fill_na_columns == 'max':
                        st.subheader("Numeric Columns Max Values")
                        st.dataframe(X_train[df_numeric].max().reset_index().rename(columns = {'index':'Columns', 0:'Columns Max Values'}))
                        X_train[df_numeric] = features[df_numeric].fillna(features[df_numeric].max())
                        X_test[df_numeric] = X_test[df_numeric].fillna(features[df_numeric].max())
                        X_val[df_numeric] = X_val[df_numeric].fillna(features[df_numeric].max())                   
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test, X_val], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test, X_val, y_val

                    elif fill_na_columns == 'mode':
                        st.subheader("Numeric Columns Mode Values")
                        df_mode = X_train[df_numeric].mode().transpose().squeeze()
                        st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))
                        X_train[df_numeric] = X_train[df_numeric].fillna(df_mode)
                        X_test[df_numeric] = X_test[df_numeric].fillna(df_mode)
                        X_val[df_numeric] = X_val[df_numeric].fillna(df_mode)
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test, X_val], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test, X_val, y_val

                    elif fill_na_columns == 'ffill':
                        features[df_numeric] = features[df_numeric].fillna(method='ffill')
                        st.subheader("Transform Numeric Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test
                    
                    elif fill_na_columns == 'bfill':
                        features[df_numeric] = features[df_numeric].fillna(method='bfill')
                        st.subheader("Transform Numeric Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test

                    elif fill_na_columns == 'interpolate':
                        interpolate_method = st.selectbox("Interpolate Method", ('linear', 'time', 'index', 'values'))
                        interpolate_direction = st.radio("Fill direction", ('Forward', 'Backward', 'Both'), horizontal=True, key='<interpolate_direction>')
                        if interpolate_method == 'time':
                            st.subheader("Transform Numeric Features")
                            st.warning('time-weighted interpolation only works on Series or DataFrames with a DatetimeIndex', icon="⚠️")
                            st.dataframe(features)
                            return features, X_train, X_test, y_train, y_test
                        else:
                            X_train[df_numeric] = X_train[df_numeric].fillna(features[df_numeric].interpolate(method=interpolate_method,limit_direction=interpolate_direction))
                            X_test[df_numeric] = X_test[df_numeric].fillna(features[df_numeric].interpolate(method=interpolate_method,limit_direction=interpolate_direction))
                            X_val[df_numeric] = X_val[df_numeric].fillna(features[df_numeric].interpolate(method=interpolate_method,limit_direction=interpolate_direction))
                            st.dataframe(X_train[df_numeric].interpolate(method=interpolate_method,limit_direction=interpolate_direction))
                            st.subheader("Transform Numeric Features")
                            features = pd.concat([X_train, X_test, X_val], axis=0)
                            st.dataframe(features)
                            return features, X_train, X_test, y_train, y_test, X_val, y_val


                    elif fill_na_columns == 0:
                        X_train[df_numeric] = X_train[df_numeric].fillna(0)
                        X_test[df_numeric] = X_test[df_numeric].fillna(0)
                        X_val[df_numeric] = X_val[df_numeric].fillna(0)                  
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test, X_val], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test, X_val, y_val
                    else:
                        pass
                else:
                     if fill_na_columns == 'mean':
                        st.subheader("Numeric Columns Mean Values")
                        st.dataframe(X_train[df_numeric].mean().reset_index().rename(columns = {'index':'Columns', 0:'Columns Mean Values'}))
                        X_train[df_numeric] = X_train[df_numeric].fillna(X_train[df_numeric].mean())
                        X_test[df_numeric] = X_test[df_numeric].fillna(X_train[df_numeric].mean())
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test], axis=0)
                       
                        st.dataframe(features)

                        return features, X_train, X_test, y_train, y_test

                     elif fill_na_columns == 'min':
                        st.subheader("Numeric Columns Min Values")
                        st.dataframe(X_train[df_numeric].min().reset_index().rename(columns = {'index':'Columns', 0:'Columns Min Values'}))
                        X_train[df_numeric] = features[df_numeric].fillna(features[df_numeric].min())
                        X_test[df_numeric] = X_test[df_numeric].fillna(features[df_numeric].min())
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test



                     elif fill_na_columns == 'max':
                        st.subheader("Numeric Columns Max Values")
                        st.dataframe(X_train[df_numeric].max().reset_index().rename(columns = {'index':'Columns', 0:'Columns Max Values'}))
                        X_train[df_numeric] = features[df_numeric].fillna(features[df_numeric].max())
                        X_test[df_numeric] = X_test[df_numeric].fillna(features[df_numeric].max())              
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test

                     elif fill_na_columns == 'mode':
                        st.subheader("Numeric Columns Mode Values")
                        df_mode = X_train[df_numeric].mode().transpose().squeeze()
                        st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))
                        X_train[df_numeric] = X_train[df_numeric].fillna(df_mode)
                        X_test[df_numeric] = X_test[df_numeric].fillna(df_mode)
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test

                     elif fill_na_columns == 'ffill':
                        features[df_numeric] = features[df_numeric].fillna(method='ffill')
                        st.subheader("Transform Numeric Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test
                    
                     elif fill_na_columns == 'bfill':
                        features[df_numeric] = features[df_numeric].fillna(method='bfill')
                        st.subheader("Transform Numeric Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test

                     elif fill_na_columns == 'interpolate':
                        interpolate_method = st.selectbox("Interpolate Method", ('linear', 'time', 'index', 'values'))
                        interpolate_direction = st.radio("Fill direction", ('Forward', 'Backward', 'Both'), horizontal=True, key='<interpolate_direction>')
                        if interpolate_method == 'time':
                            st.subheader("Transform Numeric Features")
                            st.warning('time-weighted interpolation only works on Series or DataFrames with a DatetimeIndex', icon="⚠️")
                            st.dataframe(features)
                            return features, X_train, X_test, y_train, y_test
                        else:
                            X_train[df_numeric] = X_train[df_numeric].fillna(features[df_numeric].interpolate(method=interpolate_method,limit_direction=interpolate_direction))
                            X_test[df_numeric] = X_test[df_numeric].fillna(features[df_numeric].interpolate(method=interpolate_method,limit_direction=interpolate_direction))
                            st.dataframe(X_train[df_numeric].interpolate(method=interpolate_method,limit_direction=interpolate_direction))
                            st.subheader("Transform Numeric Features")
                            features = pd.concat([X_train, X_test], axis=0)
                            st.dataframe(features)
                            return features, X_train, X_test, y_train, y_test


                     elif fill_na_columns == 0:
                        X_train[df_numeric] = X_train[df_numeric].fillna(0)
                        X_test[df_numeric] = X_test[df_numeric].fillna(0)
                        st.subheader("Transform Numeric Features")
                        features = pd.concat([X_train, X_test], axis=0)
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test
                     else:
                        pass
            # Groupby fill Na
            else:
                pass
                #X_train[df_numeric].fillna(X_train[df_numeric].mean())    
#
#
                #fill_na_columns = st.selectbox("How to FillNA",(0, 'mean','min','max','mode','interpolate'))
                #st.markdown('')
                #if validation_select == 'Yes':
                #     if fill_na_columns == 'mean':
                #        groupby_columns = st.multiselect("Select Groupby Columns", features.columns.to_list())
                #        st.subheader("Numeric Columns Mean Values")
                #        numeric_isnull =  df[df_numeric].isnull().sum()
                #        numeric_isnull_dataframe = pd.DataFrame(numeric_isnull).reset_index().rename(columns = {'index':'Columns', 0:'NA Counts'})
                #        numeric_isnull_list = numeric_isnull_dataframe.Columns[numeric_isnull_dataframe.iloc[:, 1] >= 1].to_list()
                #        
                #        st.dataframe(X_train.groupby(groupby_columns)[numeric_isnull_list].mean())
                #        
                #       
                #        grouby_df_input = X_train.groupby(groupby_columns)[numeric_isnull_list].mean()
                #        
#
                #        X_train[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.mean()))
                #        X_test[numeric_isnull_list] =X_test.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.mean()))
                #        X_val[numeric_isnull_list] =X_val.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.mean()))
#
                #        st.subheader("Transform Numeric Features")
                #        features = pd.concat([X_train, X_test, X_val], axis=0)
                #        st.dataframe(features)
                #        return features, X_train, X_test, y_train, y_test, X_val, y_val
#
                #     elif fill_na_columns == 0:
                #        X_train[df_numeric] = X_train[df_numeric].fillna(0)
                #        X_test[df_numeric] = X_test[df_numeric].fillna(0)
                #        X_val[df_numeric] = X_val[df_numeric].fillna(0)                  
                #        st.subheader("Transform Numeric Features")
                #        features = pd.concat([X_train, X_test, X_val], axis=0)
                #        st.dataframe(features)
                #        return features, X_train, X_test, y_train, y_test, X_val, y_val
#
                #     elif fill_na_columns == 'min':
                #        groupby_columns = st.multiselect("Select Groupby Columns", features.columns.to_list())
                #        st.subheader("Numeric Columns Min Values")
                #        numeric_isnull =  df[df_numeric].isnull().sum()
                #        numeric_isnull_dataframe = pd.DataFrame(numeric_isnull).reset_index().rename(columns = {'index':'Columns', 0:'NA Counts'})
                #        numeric_isnull_list = numeric_isnull_dataframe.Columns[numeric_isnull_dataframe.iloc[:, 1] >= 1].to_list()
#
                #        st.dataframe(X_train.groupby(groupby_columns)[numeric_isnull_list].min())
                #        X_train[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.min()))
                #        X_test[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.min()))
                #        X_val[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.min()))
                #        st.subheader("Transform Numeric Features")
                #        features = pd.concat([X_train, X_test, X_val], axis=0)
                #        st.dataframe(features)
#
                #     elif fill_na_columns == 'max':
                #        groupby_columns = st.multiselect("Select Groupby Columns", features.columns.to_list())
                #        st.subheader("Numeric Columns Max Values")
                #        numeric_isnull =  df[df_numeric].isnull().sum()
                #        numeric_isnull_dataframe = pd.DataFrame(numeric_isnull).reset_index().rename(columns = {'index':'Columns', 0:'NA Counts'})
                #        numeric_isnull_list = numeric_isnull_dataframe.Columns[numeric_isnull_dataframe.iloc[:, 1] >= 1].to_list()
#
                #        st.dataframe(X_train.groupby(groupby_columns)[numeric_isnull_list].max())
#
#
                #        X_train[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.max()))
                #        X_test[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.max()))
                #        X_val[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.max()))
                #        st.subheader("Transform Numeric Features")
                #        features = pd.concat([X_train, X_test, X_val], axis=0)
                #        st.dataframe(features)
#
                #     # 여기서 부터 내일 수정해야함
                #     elif fill_na_columns == 'mode':
                #        groupby_columns = st.multiselect("Select Groupby Columns", features.columns.to_list())
                #        st.subheader("Numeric Columns Mode Values")
                #        numeric_isnull =  df[df_numeric].isnull().sum()
                #        numeric_isnull_dataframe = pd.DataFrame(numeric_isnull).reset_index().rename(columns = {'index':'Columns', 0:'NA Counts'})
                #        numeric_isnull_list = numeric_isnull_dataframe.Columns[numeric_isnull_dataframe.iloc[:, 1] >= 1].to_list()
#
                #        #st.dataframe(X_train.groupby(["Region"])["Discount"].max())
                #        st.dataframe(X_train.groupby(groupby_columns)[numeric_isnull_list].apply(lambda x: x.mode()[0]))                
#
                #        X_train[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.mode()[0]))
                #        X_test[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.mode()[0]))
                #        X_val[numeric_isnull_list] =X_train.groupby(groupby_columns)[numeric_isnull_list].transform(lambda x: x.fillna(x.mode()[0]))
                #        st.subheader("Transform Numeric Features")
                #        features = pd.concat([X_train, X_test, X_val], axis=0)
                #        st.dataframe(features)
                #else:
                #    pass       

        else:   
            st.subheader("Fill Only Numeric Columns")
            st.success('There is not any NA value in your dataset.', icon="✅")

# Fill_Na(object_columns)
def fill_na_object(df):
    global features, target, X_train, X_test, y_train, y_test, X_val, y_val, fill_columns
    object_columns = df.select_dtypes(include=object).columns.to_list()
    features_isnull = df.isnull().sum().sum()
    
    
    if fill_columns == 'No':
       if features_isnull == 0:
            st.subheader("Only Object Columns")
            st.success('There is not any NA value in your dataset.', icon="✅")
            st.subheader("Features")
            st.dataframe(features)
            return features
       else:
            st.subheader("Only Object Columns")
            if df[object_columns].isnull().sum().sum() == 0:
                st.success('There is not any NA value in your dataset.', icon="✅")
            else:
                st.error("You have to Handling of Null Values", icon="🚨")
    else:
        object_isnull = df[object_columns].isnull().sum().sum()
        if object_isnull == 0:
            st.subheader("Only Object Columns")
            st.success('There is not any NA value in your dataset.', icon="✅")
            st.subheader("Features")
            st.dataframe(features)
            return features
        else:
            st.subheader("Only Object Columns")
            groupOrNongroup = st.radio("Na Values Search Method", ('Column Values', 'Groupby Values'), horizontal=True, key='<groupOrNongroup1>')
            if groupOrNongroup == 'Column Values':
                fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode','ffill', 'bfill'))
                if validation_select == 'Yes':
                
                    if fill_na_object_columns == 0:

                        X_train[df_object] = X_train[df_object].fillna(0)
                        X_test[df_object] = X_test[df_object].fillna(0)
                        X_val[df_object] = X_val[df_object].fillna(0)
                        features = pd.concat([X_train, X_test, X_val], axis=0)
                        target = pd.concat([y_train, y_test, y_val], axis=0)
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test, X_val, y_val

                    elif fill_na_object_columns == 'mode':
                        st.subheader("Object Columns Mode Values")
                        df_mode = X_train[df_object].mode().transpose()
                        df_mode = df_mode.iloc[:,:1].squeeze()
                        st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))
                        X_train[df_object] = X_train[df_object].fillna(df_mode)
                        X_test[df_object] = X_test[df_object].fillna(df_mode)
                        X_val[df_object] = X_val[df_object].fillna(df_mode)
                        features = pd.concat([X_train, X_test, X_val], axis=0)
                        target = pd.concat([y_train, y_test, y_val], axis=0)               
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test, X_val, y_val

                    elif fill_na_object_columns == 'ffill':
                        features[df_object] = features[df_object].fillna(method='ffill')
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test, y_val

                    elif fill_na_object_columns == 'bfill':
                        features[df_object] = features[df_object].fillna(method='bfill')
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test,  y_val
                else:
                    if fill_na_object_columns == 0:
    
                        X_train[df_object] = X_train[df_object].fillna(0)
                        X_test[df_object] = X_test[df_object].fillna(0)
                        features = pd.concat([X_train, X_test], axis=0)
                        target = pd.concat([y_train, y_test], axis=0)
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test

                    elif fill_na_object_columns == 'mode':
                        st.subheader("Object Columns Mode Values")
                        df_mode = X_train[df_object].mode().transpose()
                        df_mode = df_mode.iloc[:,:1].squeeze()
                        st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))
                        X_train[df_object] = X_train[df_object].fillna(df_mode)
                        X_test[df_object] = X_test[df_object].fillna(df_mode)
                        features = pd.concat([X_train, X_test], axis=0)
                        target = pd.concat([y_train, y_test], axis=0)
                 
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test

                    elif fill_na_object_columns == 'ffill':
                        features[df_object] = features[df_object].fillna(method='ffill')
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test

                    elif fill_na_object_columns == 'bfill':
                        features[df_object] = features[df_object].fillna(method='bfill')
                        st.subheader("Transform Object Features")
                        st.dataframe(features)
                        return features, X_train, X_test, y_train, y_test
                    
            else:
                pass
    
    

# 6. Label Incoder | Onehotencoder 
# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
def label_onehot_encoder(df):
    global features, X_train, X_test, y_train, y_test, X_val, y_val
    select_encoding = st.selectbox("Using Encoding?",('Yes', 'No'))
    if select_encoding == 'Yes':
        encoding_method = st.radio('select method', ('Label Encoder', 'Onehotencoder'), horizontal=True)
        if encoding_method == "Label Encoder":
            #object_columns = df.select_dtypes(include=object).columns.to_list()
            container = st.container()
            all = st.checkbox('Select all')
            if all:
                select_columns_encoding = container.multiselect('Select Columns', df_object, df_object)
                #st.info('Select Columns you want encoding', icon="ℹ️")
                if select_columns_encoding is not None:
                    le = LabelEncoder()
                    df[select_columns_encoding] = df[select_columns_encoding].apply(le.fit_transform)                                 
                    st.subheader("Transform Features")
                    st.success('Complited Encoding your Features', icon="✅")
                    st.dataframe(features)
                    return features, X_train, X_test, y_train, y_test
                else:
                    pass
            else:
                select_columns_encoding = container.multiselect('Select Columns', df_object)
                #st.info('Select Columns you want encoding', icon="ℹ️")
                if select_columns_encoding is not None:
                    le = LabelEncoder()
                    df[select_columns_encoding] = df[select_columns_encoding].apply(le.fit_transform)
                    st.subheader("Transform Features")
                    st.success('Complited Encoding your Features', icon="✅")
                    st.dataframe(features)
                    return features
                
                
        else:
            # https://datascience.stackexchange.com/questions/71804/how-to-perform-one-hot-encoding-on-multiple-categorical-columns
            # df_numeric, df_object, df_datetime
            container = st.container()
            all = st.checkbox('Select all')
            if all:
                select_columns_encoding2 = container.multiselect('Select Columns', df_object, df_object)
                if select_columns_encoding2 is not None:
                    features = pd.get_dummies(features, columns=select_columns_encoding2, prefix_sep='_', )
                    st.subheader("Transform Features")
                    st.success('Complited Encoding your Features', icon="✅")
                    st.dataframe(features)
                    return features
                else:
                    pass
            else:
                select_columns_encoding2 = container.multiselect('Select Columns', df_object)
                #st.info('Select Columns you want encoding', icon="ℹ️")
                if select_columns_encoding2 is not None:
                    features = pd.get_dummies(features, columns=select_columns_encoding2, prefix_sep='_', )
                    st.subheader("Transform Features")
                    st.success('Complited Encoding your Features', icon="✅")
                    st.dataframe(features)
                    return features
            
          
        
    else:
        
        st.subheader("No change Features")
        st.dataframe(features)      
        
#8. Numeric Columns Encoding

def numeric_columns_encoding(df):
    global features, df_numeric, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input
    
    select_numeric_encoding = st.selectbox("Using Scaler?",('Yes', 'No'))                     
    if select_numeric_encoding == 'Yes':
        scaler_method = st.radio("Select Method",('Standard', 'Nomalize', 'MinMax','MaxAbs','Robust'), horizontal=True)
        
        if validation_select == "Yes":
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size_input, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_input, random_state=42)
            if scaler_method =='Standard':
    
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                    scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                    df_scaler_columns = X_train[scaler_select_columns]
                    df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                    features = df.drop(columns=df_scaler_columns_name)

                    scaler = StandardScaler()
                    scaler = scaler.fit(df_scaler_columns)

                    X_train_scaler_data = scaler.transform(df_scaler_columns)

                    X_train_features = X_train.drop(columns=df_scaler_columns_name)
                    X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                    X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                    X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                    X_test_features = X_test.drop(columns=df_scaler_columns_name)
                    X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                    X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                    X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                    X_val_features = X_val.drop(columns=df_scaler_columns_name)
                    X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                    X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                    #scaler_df = pd.concat([X_train,X_test,X_val], axis=0)
                    #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                    features = pd.concat([X_train,X_test,X_val], axis=0).sort_index()                
                    st.dataframe(features)

                    return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 
                else:
                    scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                    df_scaler_columns = X_train[scaler_select_columns]
                    df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                    features = df.drop(columns=df_scaler_columns_name)

                    scaler = StandardScaler()
                    scaler = scaler.fit(df_scaler_columns)

                    X_train_scaler_data = scaler.transform(df_scaler_columns)

                    X_train_features = X_train.drop(columns=df_scaler_columns_name)
                    X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                    X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)
    

                    X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                    X_test_features = X_test.drop(columns=df_scaler_columns_name)
                    X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                    X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                    X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                    X_val_features = X_val.drop(columns=df_scaler_columns_name)
                    X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                    X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                    
                    #scaler_df = pd.concat((X_train,X_test,X_val), axis=0)
                    features = pd.concat([X_train,X_test,X_val], axis=0).sort_index()                                
                    st.dataframe(features)

                    return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 

            elif scaler_method == 'Nomalize':
                norm_parameter = st.selectbox('select norm method',('l2', 'l1', 'max'))
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = Normalizer(norm=norm_parameter)
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                   X_val_features = X_val.drop(columns=df_scaler_columns_name)
                   X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                   X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 
                else:
                    scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                    df_scaler_columns = X_train[scaler_select_columns]
                    df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                    features = df.drop(columns=df_scaler_columns_name)

                    scaler = Normalizer(norm=norm_parameter)
                    scaler = scaler.fit(df_scaler_columns)

                    X_train_scaler_data = scaler.transform(df_scaler_columns)

                    X_train_features = X_train.drop(columns=df_scaler_columns_name)
                    X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                    X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                    X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                    X_test_features = X_test.drop(columns=df_scaler_columns_name)
                    X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                    X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                    X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                    X_val_features = X_val.drop(columns=df_scaler_columns_name)
                    X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                    X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                    #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                    #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                    features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                    st.dataframe(features)

                    return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input

            elif scaler_method =='MinMax':
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = MinMaxScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                   X_val_features = X_val.drop(columns=df_scaler_columns_name)
                   X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                   X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 
                else:
                    scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                    df_scaler_columns = X_train[scaler_select_columns]
                    df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                    features = df.drop(columns=df_scaler_columns_name)

                    scaler = MinMaxScaler()
                    scaler = scaler.fit(df_scaler_columns)

                    X_train_scaler_data = scaler.transform(df_scaler_columns)

                    X_train_features = X_train.drop(columns=df_scaler_columns_name)
                    X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                    X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                    X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                    X_test_features = X_test.drop(columns=df_scaler_columns_name)
                    X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                    X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                    X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                    X_val_features = X_val.drop(columns=df_scaler_columns_name)
                    X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                    X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                    #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                    #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                    features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                    st.dataframe(features)

                    return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input

            elif scaler_method =='MaxAbs':
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = MaxAbsScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                   X_val_features = X_val.drop(columns=df_scaler_columns_name)
                   X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                   X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 
                else:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = MaxAbsScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                   X_val_features = X_val.drop(columns=df_scaler_columns_name)
                   X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                   X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 


            elif scaler_method =='Robust':
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = RobustScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                   X_val_features = X_val.drop(columns=df_scaler_columns_name)
                   X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                   X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 
                else:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = RobustScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   X_val_scaler_data = scaler.transform(X_val[scaler_select_columns])

                   X_val_features = X_val.drop(columns=df_scaler_columns_name)
                   X_val_scaler_data_frame = pd.DataFrame(X_val_scaler_data, columns=df_scaler_columns_name, index=X_val.index)
                   X_val = pd.concat([X_val_scaler_data_frame, X_val_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data,X_val_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test, X_val], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input 
            else:
                pass
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size_input, random_state=42)
            if scaler_method =='Standard':
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                    scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                    df_scaler_columns = X_train[scaler_select_columns]
                    df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                    features = df.drop(columns=df_scaler_columns_name)

                    scaler = StandardScaler()
                    scaler = scaler.fit(df_scaler_columns)

                    X_train_scaler_data = scaler.transform(df_scaler_columns)

                    X_train_features = X_train.drop(columns=df_scaler_columns_name)
                    X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                    X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                    X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                    X_test_features = X_test.drop(columns=df_scaler_columns_name)
                    X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                    X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                  

                    #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                    #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                    features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                    st.dataframe(features)

                    return features, features, target, df_numeric, validation_select, X_train, X_test, test_size_input
                else:
                    scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                    df_scaler_columns = X_train[scaler_select_columns]
                    df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                    features = df.drop(columns=df_scaler_columns_name)

                    scaler = StandardScaler()
                    scaler = scaler.fit(df_scaler_columns)

                    X_train_scaler_data = scaler.transform(df_scaler_columns)

                    X_train_features = X_train.drop(columns=df_scaler_columns_name)
                    X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                    X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                    X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                    X_test_features = X_test.drop(columns=df_scaler_columns_name)
                    X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                    X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)


                    #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                    #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                    features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                    st.dataframe(features)

                    return features, features, target, df_numeric, validation_select, X_train, X_test, test_size_input


            elif scaler_method =='MinMax':
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = MinMaxScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)


                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, test_size_input
                else:
                    scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                    df_scaler_columns = X_train[scaler_select_columns]
                    df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                    features = df.drop(columns=df_scaler_columns_name)

                    scaler = MinMaxScaler()
                    scaler = scaler.fit(df_scaler_columns)

                    X_train_scaler_data = scaler.transform(df_scaler_columns)

                    X_train_features = X_train.drop(columns=df_scaler_columns_name)
                    X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                    X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                    X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                    X_test_features = X_test.drop(columns=df_scaler_columns_name)
                    X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                    X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                    #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                    #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                    features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                    st.dataframe(features)

                    return features, features, target, df_numeric, validation_select, X_train, X_test, test_size_input

            elif scaler_method =='MaxAbs':
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = MaxAbsScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, test_size_input
                else:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = MaxAbsScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, y_val, test_size_input


            elif scaler_method =='Robust':
                container = st.container()
                all = st.checkbox('Select all', key='<all_encoding>')
                if all:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = RobustScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                   st.dataframe(features)

                   return features, features, target, df_numeric, validation_select, X_train, X_test, test_size_input
                else:
                   scaler_select_columns = container.multiselect('Select Columns', df_numeric, key='<scaler_select_columns>')
                   df_scaler_columns = X_train[scaler_select_columns]
                   df_scaler_columns_name = X_train[scaler_select_columns].columns.to_list()
                   features = df.drop(columns=df_scaler_columns_name)

                   scaler = RobustScaler()
                   scaler = scaler.fit(df_scaler_columns)

                   X_train_scaler_data = scaler.transform(df_scaler_columns)

                   X_train_features = X_train.drop(columns=df_scaler_columns_name)
                   X_train_scaler_data_frame = pd.DataFrame(X_train_scaler_data, columns=df_scaler_columns_name, index=X_train.index)
                   X_train = pd.concat([X_train_scaler_data_frame, X_train_features], axis=1)


                   X_test_scaler_data = scaler.transform(X_test[scaler_select_columns])

                   X_test_features = X_test.drop(columns=df_scaler_columns_name)
                   X_test_scaler_data_frame = pd.DataFrame(X_test_scaler_data, columns=df_scaler_columns_name, index=X_test.index)
                   X_test = pd.concat([X_test_scaler_data_frame, X_test_features], axis=1)

                   #scaler_df = np.concatenate((X_train_scaler_data,X_test_scaler_data), axis=0)
                   #scaler_df = pd.DataFrame(scaler_df, columns=df_scaler_columns_name)

                   features = pd.concat([X_train, X_test], axis=0).sort_index()                                
                   st.dataframe(features)
                
                   return features, features, target, df_numeric, validation_select, X_train, X_test, test_size_input
           
    else:
        st.subheader("No change Features")
        st.dataframe(features)


# 7. Create Polynomial Feature

def Polynomial_encoding(df):
    global features, target, df_numeric, validation_select, X_train, X_test, X_val, y_train, y_test, y_val, test_size_input, val_size_input
    select_polynomial = st.selectbox("Using Polynomial?",('Yes', 'No'))
    
    if select_polynomial == 'Yes':
    
        poly_select_columns = st.multiselect('Select Columns', df_numeric)
        df_poly_columns = df[poly_select_columns]
        if df_poly_columns is not None:
            number = st.slider("Degree", min_value=1, max_value=10, format='%d')
            if 0 < number < 11:
                features = df.drop(columns=df_poly_columns)                
                poly_features = PolynomialFeatures(degree=number)
                poly_data = poly_features.fit_transform(df_poly_columns)
                poly_columns_name = poly_features.get_feature_names(df_poly_columns.columns)
                poly_df = pd.DataFrame(poly_data, columns=poly_columns_name, index = features.index)               
                features =  pd.concat([poly_df, features], axis=1) # column bind  
         
                #df_poly_numeric = features.drop(columns=df_object).columns.to_list()
                
                st.subheader("Transforam Features")
                features = features.sort_index().reset_index()              
                target = target.sort_index().reset_index()

                st.dataframe(features)
              
            else:
                st.warning("please input Degree 1 ~ 10", icon="⚠️")
            
        else:
            st.warning("pleas select columns", icon="⚠️")
            
        
    else:
        st.subheader("No change Features")
        st.dataframe(features)
        #df_poly_numeric = features.drop(columns=df_object).columns.to_list() 
        
        return features, target, df_numeric, validation_select #df_poly_numeric
   
        


 
        
   
# 화면노출
st.title("🚧 Pre-processing")
upload_file = st.file_uploader("파일을 업로드 해주세요", type=['xlsx', 'xls', 'csv'], accept_multiple_files=False)

if upload_file is not None:
    try:
        # 데이터 불러오기
        df, columns = load_dataframe(upload_file=upload_file)
        st.markdown("---")
        st.subheader("#1. LOAD DATA")
        try:
            st.dataframe(df) # 함수
        except Exception as e:
            print(e)

        st.markdown("---")
        st.subheader("#2. Drop Columns")
        try:
            drop_df(df)
        except Exception as e:
            print(e)
            
        st.markdown("---")
        st.subheader("#3. DropNa")
        try:
            Drop_na(features)
        except Exception as e:
            print(e)    
    
        st.markdown("---")
        st.subheader("#4. Split Features & Target")
        try:
            
            split_x_y(features)
        except Exception as e:
            print(e)

        
        st.markdown("---")
        st.subheader("#5. Train & Test & Val Split")
        try:
            split_train_test_split(features, target)
        except Exception as e:
            print(e)


        # drop outlier는 타겟 데이터랑 같이 삭제되야함.   
        
        #st.markdown("---")
        #st.subheader("#6.Drop Outlier")
        #try: 
        #    fill_na(features)
        #    fill_na_object(features)
        #except Exception as e:
        #    print(e)
            
        st.markdown("---")
        st.subheader("#7.Fill NA")
        try: 
            fill_na(features)
            fill_na_object(features)
            
        except Exception as e:
            print(e)        

            
        st.markdown("---")
        st.subheader("#8. Object Columns Encoding")
        try:
            label_onehot_encoder(features)
        except Exception as e:
            print(e)
            
        st.markdown("---")
        st.subheader("#9. Numeric Columns Encoding")
        try:    
            numeric_columns_encoding(features)
        except Exception as e:
            print(e)    
                      
        st.markdown("---")
        st.subheader("#10. Polynomial Features")
        try:    
            Polynomial_encoding(features)
    
        except Exception as e:
            print(e)
                    
     
    except Exception as e:
        print(e)
        
        
        


else:
    pass
