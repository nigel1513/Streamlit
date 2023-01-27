import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
from PIL import Image
import base64 # Standard Python Module 
from io import StringIO, BytesIO # Standard Python Module
import streamlit.components.v1 as components


# set_page_confit
st.set_page_config(page_title='EDA', page_icon="🔎")

# Page_title
st.title('🔎 Exploratory data analysis')
upload_file = st.file_uploader(label="",
                accept_multiple_files=False,
                type=['csv', 'xlsx', 'xls'])



#background color setting CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #ffffff;
opacity: 1;
background-image:  radial-gradient(#000000 0.75px, transparent 0.75px), radial-gradient(#000000 0.75px, #ffffff 0.75px);
background-size: 30px 30px;
background-position: 0 0,15px 15px;
}
[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}
</style>
"""
#st.markdown(page_bg_img, unsafe_allow_html=True)


# function definition

#-- 1.Data Load
@st.cache
def load_dataframe():
    try:
        df = pd.read_csv(upload_file)
        
    except Exception as e:
        print(e)
        df = pd.read_excel(upload_file)

    columns = list(df.columns)
    columns.append(None)

    return df, columns  #df, columns =columns_list 

#-- 2. Data Info
buffer = io.StringIO()
def data_info(df):
    try:
        df_shape = df.shape
        df_info = df.info(buf=buffer)
        s = buffer.getvalue()
        st.write("👁‍🗨Data Shape:", df_shape)
        st.text(s)
        st.markdown("---")
        
    except Exception as e:
        print(e)    

#-- 3. Data Describe
def data_describe(df):
    try:
        df_describe = df.describe()
        st.write(df_describe)
    except Exception as e:
        print(e)

#-- 4. Data Missing isnull
def df_isnull(df):
     if df.isnull().sum().sum() == 0:
        st.success('There is not any NA value in your dataset.', icon="✅")
     else:
        try:
            isnull = pd.DataFrame(df.isnull().sum()).reset_index()
            isnull['Percentage'] = round(isnull[0] / df.shape[0] * 100, 2)
            isnull['Percentage'] = isnull['Percentage'].astype(str) + '%'
            isnull = isnull.rename(columns = {'index':'Column', 0:'Number of null values'})
            st.dataframe(isnull)
            return isnull
        except Exception as e:
            print(e)
# --5. isnull plot

def df_isnull_plot(df):
    fig = px.bar(df,
             x= 'Column',
             y='Number of null values',
             text ='Column',
             barmode='stack')
    fig.update_layout( legend={'title': None})
    title={'text':"Null Values Ratio",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'}
    fig.update_traces(textposition="outside")
    fig.update_layout(title=title)
    st.plotly_chart(fig)

#-- histogram

def histogram_plot(df):
    try:
        df_numeric = df.select_dtypes(exclude = 'object').columns.to_list()
        df_non_numeric = df.select_dtypes(include = 'object').columns.to_list()
        df_non_numeric.insert(0,None)
        hitogram_plot_column = st.selectbox("Select Column", df_numeric, key = "<uniquevalueofsomesort1>")
        color_value = st.selectbox("Color", df_non_numeric, key = "<uniquevalueofsomesort2>")

        bin_size = st.slider("Bins", min_value=10, max_value=100)
        fig = px.histogram(df, x=hitogram_plot_column, nbins=bin_size, color=color_value)
        fig.update_layout(
            title={
            'text': "Histogram",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
        st.plotly_chart(fig)
    except Exception as e:
        print(e)


#-- 5. space function
def space(num_lines=1):
    for _ in range(num_lines):
        st.write("")

#-- 6. Outlier function
def df_outlier(df):
    try:
        df_outlier = df.select_dtypes(exclude = ['object','datetime'])

        q1 = df_outlier.quantile(0.25)
        q3 = df_outlier.quantile(0.75)
        iqr = q3 - q1


        lower_range_count = (df_outlier < q1 - (1.5 * iqr)).sum()
        lower_range_count = pd.DataFrame(lower_range_count).reset_index().rename(columns = {'index':'column', 0:'OutlierLower(count)'})

        upper_range_count = (df_outlier > q3 + (1.5 * iqr)).sum()
        upper_range_count = pd.DataFrame(upper_range_count).reset_index().rename(columns = {'index':'column', 0:'OutlierUpper(count)'})

        lower_range = q1 - (1.5 * iqr)
        lower_range = pd.DataFrame(lower_range).reset_index().rename(columns = {'index':'column', 0:'LowerRange'})

        upper_range = q3 + (1.5 * iqr)
        upper_range = pd.DataFrame(upper_range).reset_index().rename(columns = {'index':'column', 0:'UpperRange'})

        ans = ((df_outlier < (q1 - 1.5 * iqr)) | (df_outlier > (q3 + 1.5 * iqr))).sum()
        iqr = pd.DataFrame(iqr).reset_index().rename(columns = {'index':'column', 0:'IQR'})
        df_outlier = pd.DataFrame(ans).reset_index().rename(columns = {'index':'column', 0:'count_of_outliers'})
        df_outlier = pd.merge(df_outlier, iqr)
        df_outlier = pd.merge(df_outlier, lower_range)
        df_outlier = pd.merge(df_outlier, lower_range_count)
        df_outlier = pd.merge(df_outlier, upper_range)
        df_outlier = pd.merge(df_outlier, upper_range_count)
        df_outlier['Percentage'] = round(df_outlier['count_of_outliers'] / df.shape[0] * 100, 2)
        df_outlier['Percentage'] = df_outlier['Percentage'].astype(str) + '%'
        return df_outlier
    except Exception as e:
        print(e)
        
# --7. outlier plot

def df_outlier_plot(df):
    fig = px.bar(df,
             x='column',
             y='count_of_outliers',
             text ='column',
             barmode='stack')
    fig.update_layout( legend={'title': None})
    title={'text':"Outlier Ratio",
    'y':0.95,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'}
    fig.update_traces(textposition="outside")
    fig.update_layout(title=title)
    st.plotly_chart(fig)

# --8. outlier violin plot

def df_violin_plot(df):
    try:
        df_numeric = df.select_dtypes(exclude = 'object').columns.to_list()
        df_non_numeric = df.select_dtypes(include = 'object').columns.to_list()
        df_non_numeric.insert(0,None)
        st.info('Boxplot select first Y_column', icon="ℹ️")
        df_non_numeric_y = st.selectbox("Select X_Column", df_non_numeric, key = "<uniquevalueofsomesort4>")
        df_numeric = st.selectbox("Select Y_Column", df_numeric, key = "<uniquevalueofsomesort3>")
        color_value = st.selectbox("Color", df_non_numeric, key = "<uniquevalueofsomesort5>")
        fig = px.violin(df, x=df_non_numeric_y, y=df_numeric, box=True, color=color_value,points='all')
        fig.update_layout(
            title={
            'text': "Violin Plot",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
        st.plotly_chart(fig)
    except Exception as e:
        print(e)

# --9. Scatter plot
def df_scatter_plot(df):
    try:
        st.info('If you want ols', icon="ℹ️")
        df_numeric = df.select_dtypes(exclude = 'object').columns.to_list()
        df_non_numeric = df.select_dtypes(include = 'object').columns.to_list()
        df_non_numeric.insert(0,None)

        X_column = st.selectbox("Select X_Column", df_numeric, key = "<uniquevalueofsomesort6>")
        Y_column = st.selectbox("Select Y_Column", df_numeric, key = "<uniquevalueofsomesort7>")
        color_value = st.selectbox("Color", df_non_numeric, key = "<uniquevalueofsomesort8>")
        if color_value is not None:
            fig = px.scatter(df, x=X_column, y=Y_column, color=color_value)
            fig.update_layout(
            title={
            'text': "Scatter Plot",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
            st.plotly_chart(fig)
        else:
            fig = px.scatter(df, x=X_column, y=Y_column, color=color_value, trendline="ols")
            fig.update_layout(
            title={
            'text': "Scatter Plot",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
            st.plotly_chart(fig)
    except Exception as e:
        print(e)

# --10 Corr & heatmap
def df_corr(df, method1):
    try:
        df_corr = df.corr(method=method1)
        st.write(df_corr)
        mask = np.zeros_like(df_corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        heat_map_fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df_corr, 
                    cmap="RdYlBu_r",
                    ax=ax,
                    mask = mask,
                    annot=True, 
                    linewidth=0.5, 
                    cbar_kws={"shrink": 1}, 
                    vmin= -1, vmax= 1
                    )
        ax.set_title(f'{method1} Heatmap')         
        st.pyplot(heat_map_fig)
    except Exception as e:
        print(e)

 # --11 value_cout
def df_value_counts(df):
    try:
        value_filtering = df.select_dtypes(include=object).columns.to_list()
        value_filtering_len = len(value_filtering)
        if value_filtering_len !=0:
            value_columns = st.selectbox("Select Column", value_filtering, key ="<uniquevalueofsomesort15>")
            df_value_counts = df[value_columns].value_counts().reset_index(name='counts')
            df_value_counts['Percentage'] = np.round((df_value_counts.counts / df_value_counts.counts.sum()) * 100, 2)
            df_value_counts['Percentage'] = df_value_counts['Percentage'].astype(str) + '%'
            st.dataframe(df_value_counts)

            fig = px.bar(df_value_counts,
                 x='index',
                 y='counts',
                 text ='index',
                 barmode='stack')
            fig.update_layout( legend={'title': None})
            title={'text':"Value Counts",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
            fig.update_traces(textposition="outside")
            fig.update_layout(title=title)
            st.plotly_chart(fig)
        else:
            st.success('There is not any Object Column in your dataset.', icon="✅")    
    except Exception as e:
        print(e)

# --11 Pivot_table
def df_pivot_table(df):
    try:
        filtering = list(df.columns)
        nonobject_filtering = df.select_dtypes(exclude = 'object').columns.to_list()
        agg = ['count', 'sum', 'sem', 'skew', 'mean', 'min', 'std', 'var', 'mad', 'max', 'quantile', 'nunique', 'size']
        filter_select = st.multiselect("Select Index", filtering, key='<filter_select>')
        filter_select2 = st.multiselect("Select Column", filtering, key='<filter_select2>')
        filter_select3 = st.multiselect("Select Values", nonobject_filtering, key='<filter_select3>')

        agg_select = st.selectbox("Aggrigation Select", agg)
        if agg_select == 'mad':
            st.info('mad: mean average deviation', icon="ℹ️")
        elif agg_select == 'sem':
            st.info('sem: standard error of mean', icon="ℹ️")
        else:
            pass

        df_pivot_table = pd.pivot_table(data=df , index=filter_select, values=filter_select3,columns=filter_select2, aggfunc=agg_select, fill_value = 0,
                         margins=True)
        st.dataframe(df_pivot_table)
    except Exception as e:
        print(e)
        

# 노출화면 함수 넣어

if upload_file is not None:
        # 데이터 불러오기
        df, columns = load_dataframe(upload_file=upload_file)
        st.markdown("---")
        st.subheader("#1. RAW DATA")
        st.dataframe(df.iloc[:,1:]) # 함수

  
        # 데이터 info
        st.markdown("---")
        st.subheader("#2. DATA INFO & SHAPE")
        data_info(df.iloc[:,1:]) # 함수
        
        
        # 데이터 Describe
        
        st.subheader("#3 DATA Describe")
        data_describe(df.iloc[:,1:])

        # 데이터 histogram
        st.markdown("---")
        st.subheader("#4 DATA HISTOGRAM")
        #df_numeric = df.select_dtypes(exclude = 'object').columns.to_list()
        try:
            histogram_plot(df)
        except Exception as e:
            print(e)

        # 데이터 Missing value / visualization
        st.markdown("---")
        st.subheader("#5 DATA MISSING VALUE")
        try:
            df_isnull = df_isnull(df.iloc[:,1:])
            #st.dataframe(df_isnull, width=800)
            df_isnull_plot(df_isnull)
        except Exception as e:
            print(e)

        # 데이터 outlier
        st.markdown("---")
        st.subheader("#6 DATA OUTLIER")
        df_outlier = df_outlier(df.iloc[:,1:])
        if df_outlier['count_of_outliers'].sum() != 0:
            try:
                #df_outlier = df_outlier(df.iloc[:,1:])
                st.dataframe(df_outlier)
                df_outlier_plot(df_outlier)
            except Exception as e:
                print(e)
        else:
            st.success('There is not any Outlier value in your dataset.', icon="✅")

        # 데이터 violin
        st.markdown("---")
        st.subheader("#7 OUTLIER VISUALIZATION")
        if df_outlier['count_of_outliers'].sum() !=0:
            try:
                df_violin_plot(df)
            except Exception as e:
                print(e)
        else:
            st.success('There is not any Outlier value in your dataset.', icon="✅")

        # 데이터 outlier
        st.markdown("---")
        st.subheader("#8 Scatter Plot")
        try:
            df_scatter_plot(df)
        except Exception as e:
            print(e)

        # 데이터 correlation
        st.markdown("---")
        st.subheader("#9 Correlation")
        tab1, tab2, tab3 = st.tabs(["Pearson", "Kendall", "Spearman"])

        with tab1:
            df_corr(df, 'pearson')

        with tab2:
           df_corr(df, 'kendall')

        with tab3:
           df_corr(df, 'spearman')

        # 데이터 value_counts
        st.markdown("---")
        st.subheader("#10 Unique Column Counts")
        try:
            df_value_counts(df)
        except Exception as e:
            print(e)

        # 데이터 Pivot table
        st.markdown("---")
        st.subheader("#11 Pivot Table")
        try:
            df_pivot_table(df)
        except Exception as e:
            print(e)

        #------------ 나중에 새창열기로 만들기 -----------
        #t = pivot_ui(df)

        #with open(t.src, encoding='utf8') as t:
           #components.html(t.read(), width=800, height=1000, scrolling=True)
        #------------ 나중에 새창열기로 만들기 -----------


