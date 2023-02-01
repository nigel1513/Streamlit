import streamlit as st
import pandas as pd
import matplotlib
import asyncio
import os
import aiohttp
from PIL import Image

from pycaret.regression import *
from pycaret.classification import *
from autoviz.AutoViz_Class import AutoViz_Class
import streamlit.components.v1 as components


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

def listdir(dir):
    filenames = os.listdir(dir)
    for filename in filenames:
        path_to_html = os.path.join(folder_path, filename)
        file_name = filename.split('.')[0]
        with open(path_to_html,'r') as f: 
            html_data = f.read()
        st.markdown("---")
        st.subheader(file_name)    
        st.components.v1.html(html_data, width=1000, height=500, scrolling=True) 

# 화면 노출

st.title("🤖 Auto-ML (PyCaret)")
upload_file = st.file_uploader("",type=['xlsx', 'xls', 'csv'], accept_multiple_files=False)
try:
    if upload_file is not None:
        st.header("#1. Raw Data")
        df, columns = load_dataframe(upload_file=upload_file)
        st.dataframe(df)
        target_select = st.selectbox("Select Target", columns)

        st.markdown("---")
        st.header("#2. Classification / Regression")
        select_method = st.selectbox("select", (None, 'Classification', 'Regression'))


        if select_method == 'Regression':    

            from pycaret.regression import *
            exp_name = setup(data= df, target =target_select, html=False, silent=True, normalize=True, transformation=True, transform_target=True,
                             combine_rare_levels=True, rare_level_threshold= 0.05, remove_multicollinearity=True, multicollinearity_threshold=0.95,
                             log_experiment=True, experiment_name='diamond1')
            #handle_unknown_categorical= True, unknown_categorical_method='most_frequent', remove_outliers=True,outliers_threshold=0.05, feature_interaction = True, feature_ratio = True)    
            contents = pull()
            contents = contents.data
            contents = contents.astype(str)
            st.markdown("---")
            st.header("#3. PyCaret Setup")
            st.dataframe(contents)

            st.markdown("---")
            st.header("#4. Transform Data")
            transform_df = pd.concat([get_config('X_train'), get_config('X_test')])
            st.dataframe(transform_df)

            dfte = eda(display_format = 'html')
            st.markdown("---")
            st.header("#5. PyCaret EDA Plot")

            # 파일 경로
            path = os.path.dirname(os.path.abspath('cat_var_plots.html')) 
            folder_path = os.path.join(path,'AutoViz_Plots', target_select)
            listdir(folder_path)

            st.markdown("---")
            st.header("#6. Best model")
            best_model = compare_models()
            best_model1 = pull()
            best_model1 = pd.DataFrame(best_model1)
            st.dataframe(best_model1)
            best_model_name = best_model1.iloc[0,0]

            st.markdown("---")
            best_model_index = best_model1.index[0]
            st.header("#7. Create Single model:" + best_model_name)
            best_single_model = create_model(best_model_index)
            best_single_model1 = pull()
            best_single_model1 = best_single_model1.reset_index()
            best_single_model1['Fold'] = best_single_model1['Fold'].astype(str)
            st.dataframe(best_single_model1)

            st.markdown("---")
            st.header("#8. Tune Model")
            tune_single_model = tune_model(best_single_model) # n_iter 값 수정 필요
            tune_single_model1 = pull()
            tune_single_model1 = tune_single_model1.reset_index()
            tune_single_model1['Fold'] = tune_single_model1['Fold'].astype(str)
            st.dataframe(tune_single_model1)

            try:
                st.header("#9. Residuals Plot") 
                residual_plot = plot_model(tune_single_model, plot ='residuals', display_format='streamlit')
                #residual_interactive_plot = plot_model(tune_single_model, plot ='residuals_interactive', save=True)
            except Exception as e:
                print(e)    

            try:
                st.header("#10. Prediction Error Plot") 
                error_plot = plot_model(tune_single_model, plot ='error', display_format='streamlit')
            except Exception as e:
                print(e)    

            try:
                st.header("#11. Cook's Distance Plot") 
                cooks_plot = plot_model(tune_single_model, plot ='cooks', display_format='streamlit')
            except Exception as e:
                print(e) 

            #try:
            #    st.header("Rfe Plot")
            #    rfe_plot = plot_model(tune_single_model, plot ='rfe', save=True)
            #except Exception as e:
            #    print(e) 

            try:
                st.header("#12. Learning Curve Plot") 
                learning_plot = plot_model(tune_single_model, plot ='learning', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("#13. Validation Curve Plot") 
                vc_plot = plot_model(tune_single_model, plot ='vc', display_format='streamlit')
                #manifold_plot = plot_model(tune_single_model, plot ='manifold', save=True)
            except Exception as e:
                print(e)

            try:
                st.header("#14. Feature Importance Plot") 
                feature_plot = plot_model(tune_single_model, plot ='feature', display_format='streamlit')
                #feature_all_plot = plot_model(tune_single_model, plot ='feautre_all', save=True)
            except Exception as e:
                print(e)

            try:
                st.header("#14. Parameter Values") 
                parameter_plot = plot_model(tune_single_model, plot ='parameter', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Tree Plot")
                tree_plot = plot_model(tune_single_model, plot ='tree', display_format='streamlit')
            except Exception as e:
                print(e)


            if best_model_index == 'rf' or 'et' or 'dt' or 'xgboost' or 'lightgbm':
                try:
                    interpret_model_chart = interpret_model(tune_single_model, save=True)        
                    path = os.path.dirname(os.path.abspath('SHAP summary.png')) 
                    folder_path = os.path.join(path, 'SHAP summary.png')
                    image = Image.open(folder_path)
                    st.markdown("---")
                    st.header("#9.Model Agnostic(Summary)")     
                    st.image(image)
                except Exception as e:
                    print(e)

                try:
                    interpret_model_chart_corr = interpret_model(tune_single_model, plot= 'correlation', save=True)        
                    path1 = os.path.dirname(os.path.abspath('SHAP correlation.png')) 
                    folder_path1 = os.path.join(path1, 'SHAP correlation.png')
                    image1 = Image.open(folder_path1)
                    st.markdown("---")
                    st.header("#10. Model Correlation Plot")     
                    st.image(image1)
                except Exception as e:
                    print(e)

                try:
                    interpret_model_chart_reason = interpret_model(tune_single_model, plot = 'reason', observation = 0, save=True)
                    path2 = os.path.dirname(os.path.abspath('SHAP reason.html')) 
                    folder_path2 = os.path.join(path2, 'SHAP reason.html')
                    with open(folder_path2, 'r', encoding='UTF8') as f: 
                        html_data2 = f.read()

                    st.markdown("---")
                    st.header("#11. Model Reason Plot") 
                    st.components.v1.html(html_data2, width=1000, height=500, scrolling=True) 
                except Exception as e:
                    print(e)            

            else:
                pass
        
        

        
                
        elif select_method == 'Classification':  

            from pycaret.classification import *
            exp_name = setup(data= df, target =target_select, html=False, silent=True)
            # handle_unknown_categorical= True, unknown_categorical_method='most_frequent', fix_imbalance=True, remove_outliers=True, outliers_threshold=0.05)
            contents = pull()
            contents = contents.data
            contents = contents.astype(str)
            st.markdown("---")
            st.header("#3. PyCaret Setup")
            st.dataframe(contents)

            st.markdown("---")
            st.header("#4. Transform Data")
            transform_df = pd.concat([get_config('X_train'), get_config('X_test')])
            st.dataframe(transform_df)

            dfte = eda(display_format = 'html')
            st.markdown("---")
            st.header("#5. PyCaret EDA Plot")

            # 파일 경로
            path = os.path.dirname(os.path.abspath('cat_var_plots.html')) 
            folder_path = os.path.join(path,'AutoViz_Plots', target_select)    
            listdir(folder_path)


            st.markdown("---")
            st.header("#6. Best model")
            best_model = compare_models()
            best_model1 = pull()
            best_model1 = pd.DataFrame(best_model1)
            st.dataframe(best_model1)
            best_model_name = best_model1.iloc[0,0]

            st.markdown("---")
            best_model_index = best_model1.index[0]
            st.header("#7. Create Single model:" + best_model_name)
            best_single_model = create_model(best_model_index)
            best_single_model1 = pull()
            best_single_model1 = best_single_model1.reset_index()
            best_single_model1['Fold'] = best_single_model1['Fold'].astype(str)
            st.dataframe(best_single_model1)

            st.markdown("---")
            st.header("#8. Tune Model")
            tune_single_model = tune_model(best_single_model, n_iter=20, early_stopping=True)
            tune_single_model1 = pull()
            tune_single_model1 = tune_single_model1.reset_index()
            tune_single_model1['Fold'] = tune_single_model1['Fold'].astype(str)
            st.dataframe(tune_single_model1)

            try:
                st.header("#9. Auc Courve") 
                Auc_plot = plot_model(tune_single_model, plot ='auc', display_format='streamlit')
                #residual_interactive_plot = plot_model(tune_single_model, plot ='residuals_interactive', save=True)
            except Exception as e:
                print(e)    

            try:
                st.header("#10. Threshold") 
                Thresh_plot = plot_model(tune_single_model, plot ='threshold', display_format='streamlit')
            except Exception as e:
                print(e)    

            try:
                st.header("#11. Precision Recall") 
                precision_plot = plot_model(tune_single_model, plot ='pr', display_format='streamlit')
            except Exception as e:
                print(e) 

            try:
                st.header("Confusion Matrix")
                confusion_plot = plot_model(tune_single_model, plot ='confusion_matrix', save=True)
            except Exception as e:
                print(e) 

            try:
                st.header("#12. Error Plot") 
                error_plot = plot_model(tune_single_model, plot ='error', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("#13. Classification Report") 
                classification_plot = plot_model(tune_single_model, plot ='class_report', display_format='streamlit')
                #manifold_plot = plot_model(tune_single_model, plot ='manifold', save=True)
            except Exception as e:
                print(e)


            try:
                st.header("#14. Boundary") 
                boundary_plot = plot_model(tune_single_model, plot ='boundary', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Learning")
                tree_plot = plot_model(tune_single_model, plot ='learning', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Calibration Curve")
                calibration_plot = plot_model(tune_single_model, plot ='calibration', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Validation Curve")
                valid_plot = plot_model(tune_single_model, plot ='vc', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Feature importance")
                feature_plot = plot_model(tune_single_model, plot ='feature', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Parameter Values") 
                parameter_plot = plot_model(tune_single_model, plot ='parameter', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Lift Curve") 
                lift_plot = plot_model(tune_single_model, plot ='lift', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("Gain Chart") 
                lift_plot = plot_model(tune_single_model, plot ='gain', display_format='streamlit')
            except Exception as e:
                print(e)

            try:
                st.header("KS plot") 
                lift_plot = plot_model(tune_single_model, plot ='ks', display_format='streamlit')
            except Exception as e:
                print(e)


            if best_model_index == 'rf' or 'et' or 'dt' or 'xgboost' or 'lightgbm':
                interpret_model_chart = interpret_model(tune_single_model, save=True)        
                path = os.path.dirname(os.path.abspath('SHAP summary.png')) 
                folder_path = os.path.join(path, 'SHAP summary.png')
                image = Image.open(folder_path)
                st.markdown("---")
                st.header("#9.Model Agnostic(Summary)")     
                st.image(image)

                interpret_model_chart_corr = interpret_model(tune_single_model, plot= 'correlation', save=True)        
                path1 = os.path.dirname(os.path.abspath('SHAP correlation.png')) 
                folder_path1 = os.path.join(path1, 'SHAP correlation.png')
                image1 = Image.open(folder_path1)
                st.markdown("---")
                st.header("#10. Model Correlation Plot")     
                st.image(image1)


                interpret_model_chart_reason = interpret_model(tune_single_model, plot = 'reason', observation = 0, save=True)
                path2 = os.path.dirname(os.path.abspath('SHAP reason.html')) 
                folder_path2 = os.path.join(path2, 'SHAP reason.html')
                with open(folder_path2, 'r', encoding='UTF8') as f: 
                    html_data2 = f.read()

                st.markdown("---")
                st.header("#11. Model Reason Plot") 
                st.components.v1.html(html_data2, width=1000, height=500, scrolling=True)
            else:
                pass
except Exception as e:
    print(e)        
else:
     pass
