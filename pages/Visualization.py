import streamlit as st
import plotly.express as px
from base64 import b64encode
import io
import pandas as pd




st.set_page_config(
    page_title='visualization')



st.title("📊Data Visualization")
upload_file = st.file_uploader(label="",
                accept_multiple_files=False,
                type=['csv', 'xlsx', 'xls'])

st.subheader('\n')

@st.cache
def download_chart(plot, output_format):
    """
    :param plot: plotly figure
    :param output_format: str, the required output format in string
    :return:
    """

    file_name_with_extension = 'image1' + output_format

    if output_format == '.html':
        buffer = io.StringIO()
        plot.write_html(buffer)
        html_bytes = buffer.getvalue().encode()
        encoding = b64encode(html_bytes).decode()

        href = f'<a download={file_name_with_extension} href="data:file/html;base64,{encoding}" >Download</a>'

    if output_format == '.json':
        img_bytes = plot.to_image(format='json', engine="auto")
        encoding = b64encode(img_bytes).decode()

        href = f'<a download={file_name_with_extension} href="data:file/json;base64,{encoding}" >Download</a>'

    if output_format == '.png':
        img_bytes = plot.to_image(format='png', engine="auto")
        encoding = b64encode(img_bytes).decode()

        href = f'<a download={file_name_with_extension} href="data:image/png;base64,{encoding}" >Download</a>'

    if output_format == '.jpeg':
        img_bytes = plot.to_image(format='jpg', engine="auto")
        encoding = b64encode(img_bytes).decode()

        href = f'<a download={file_name_with_extension} href="data:image/jpeg;base64,{encoding}" >Download</a>'

    if output_format == '.svg':
        img_bytes = plot.to_image(format='svg', engine="auto")
        encoding = b64encode(img_bytes).decode()

        href = f'<a download={file_name_with_extension} href="data:image/svg;base64,{encoding}" >Download</a>'

    if output_format == '.pdf':
        img_bytes = plot.to_image(format='pdf', engine="auto")
        encoding = b64encode(img_bytes).decode()

        href = f'<a download={file_name_with_extension} href="data:file/pdf;base64,{encoding}" >Download</a>'

    return href



def show_export_format(plot):
    try:
        st.subheader('Download')
        output_format = st.selectbox(label='File Format', options=['.png', '.jpeg', '.pdf', '.svg',
                                                                              '.html', '.json'])
        href = download_chart(plot, output_format=output_format)
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        print(e)



def graph_controls(chart_type, df, dropdown_options, template):
    """
    Function which determines the widgets that would be shown for the different chart types
    :param chart_type: str, name of chart
    :param df: upload dataframe
    :param dropdown_options: list of column names
    :param template: str, representation of the selected theme
    :return:
    """
    length_of_options = len(dropdown_options)
    length_of_options -= 1

    plot = px.scatter()

    if chart_type == 'Scatter plots':
        st.sidebar.subheader("🔨plot Settings")

        try:
            x_values = st.sidebar.selectbox('X axis', index=length_of_options,options=dropdown_options)
            y_values = st.sidebar.selectbox('Y axis',index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)
            symbol_value = st.sidebar.selectbox("Symbol",index=length_of_options, options=dropdown_options)
            size_value = st.sidebar.selectbox("Size", index=length_of_options,options=dropdown_options)
            hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options,options=dropdown_options)
            facet_row_value = st.sidebar.selectbox("Facet row",index=length_of_options, options=dropdown_options,)
            facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
                                                      options=dropdown_options)
            marginalx = st.sidebar.selectbox("Marginal X", index=2,options=['rug', 'box', None,
                                                                         'violin', 'histogram'])
            marginaly = st.sidebar.selectbox("Marginal Y", index=2,options=['rug', 'box', None,
                                                                         'violin', 'histogram'])
            log_x = st.sidebar.selectbox('Log axis on x', options=[True, False])
            log_y = st.sidebar.selectbox('Log axis on y', options=[True, False])
            #title = st.text_input(label='Title of chart')
            plot = px.scatter(data_frame=df,
                              x=x_values,
                              y=y_values,
                              color=color_value,
                              symbol=symbol_value,
                              size=size_value,
                              hover_name=hover_name_value,
                              facet_row=facet_row_value,
                              facet_col=facet_column_value,
                              log_x=log_x, log_y=log_y,marginal_y=marginaly, marginal_x=marginalx,
                              template=template) #title = title 

        except Exception as e:
            print(e)

    if chart_type == 'Histogram':
        st.sidebar.subheader("🔨plot Settings")

        try:
            x_values = st.sidebar.selectbox('X axis', index=length_of_options,options=dropdown_options)
            y_values = st.sidebar.selectbox('Y axis',index=length_of_options, options=dropdown_options)
            nbins = st.sidebar.number_input(label='Number of bins', min_value=2, value=5)
            color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)

            barmode = st.sidebar.selectbox('bar mode', options=['group', 'overlay','relative'], index=2)
            marginal = st.sidebar.selectbox("Marginal", index=2,options=['rug', 'box', None,
                                                                         'violin', 'histogram'])
            barnorm = st.sidebar.selectbox('Bar norm', options=[None, 'fraction', 'percent'], index=0)
            hist_func = st.sidebar.selectbox('Histogram aggregation function', index=0,
                                             options=['count','sum', 'avg', 'min', 'max'])
            histnorm = st.sidebar.selectbox('Hist norm', options=[None, 'percent', 'probability', 'density',
                                                                  'probability density'], index=0)
            hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options,options=dropdown_options)
            facet_row_value = st.sidebar.selectbox("Facet row",index=length_of_options, options=dropdown_options,)
            facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
                                                      options=dropdown_options)
            cummulative = st.sidebar.selectbox('Cummulative', options=[False, True])
            log_x = st.sidebar.selectbox('Log axis on x', options=[True, False])
            log_y = st.sidebar.selectbox('Log axis on y', options=[True, False])
            #title = st.text_input(label='Title of chart')
            plot = px.histogram(data_frame=df,barmode=barmode,histnorm=histnorm,
                                marginal=marginal,barnorm=barnorm,histfunc=hist_func,
                                x=x_values,y=y_values,cumulative=cummulative,
                                color=color_value,hover_name=hover_name_value,
                                facet_row=facet_row_value,nbins=nbins,
                                facet_col=facet_column_value,log_x=log_x,
                                log_y=log_y,template=template) #title=title

        except Exception as e:
            print(e)

    # if chart_type == 'Line plots':
    #     st.subheader("Line plots Settings")
    #
    #     try:
    #         x_values = st.selectbox('X axis', index=length_of_options, options=dropdown_options)
    #         y_values = st.selectbox('Y axis', options=dropdown_options)
    #         color_value = st.selectbox("Color", index=length_of_options, options=dropdown_options)
    #         line_group = st.selectbox("Line group", options=dropdown_options)
    #         line_dash = st.selectbox("Line dash", index=length_of_options,options=dropdown_options)
    #         hover_name_value = st.selectbox("Hover name", index=length_of_options, options=dropdown_options)
    #         facet_row_value = st.selectbox("Facet row", index=length_of_options, options=dropdown_options, )
    #         facet_column_value = st.selectbox("Facet column", index=length_of_options,
    #                                                   options=dropdown_options)
    #         log_x = st.selectbox('Log axis on x', options=[True, False])
    #         log_y = st.selectbox('Log axis on y', options=[True, False])
    #         title = st.text_input(label='Title of chart')
    #         plot = px.line(data_frame=df,
    #                        line_group=line_group,
    #                        line_dash=line_dash,
    #                        x=x_values,y=y_values,
    #                        color=color_value,
    #                        hover_name=hover_name_value,
    #                        facet_row=facet_row_value,
    #                        facet_col=facet_column_value,
    #                        log_x=log_x,
    #                        log_y=log_y,
    #                        template=template,
    #                        title=title)
    #     except Exception as e:
    #         print(e)

    if chart_type == 'Violin plots':
        st.sidebar.subheader('🔨plot Settings')

        try:
            x_values = st.sidebar.selectbox('X axis', index=length_of_options,options=dropdown_options)
            y_values = st.sidebar.selectbox('Y axis',index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)
            violinmode = st.sidebar.selectbox('Violin mode', options=['group', 'overlay'])
            box = st.sidebar.selectbox("Show box", options=[False, True])
            outliers = st.sidebar.selectbox('Show points', options=[False, 'all', 'outliers', 'suspectedoutliers'])
            hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options,options=dropdown_options)
            facet_row_value = st.sidebar.selectbox("Facet row",index=length_of_options, options=dropdown_options,)
            facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
                                                      options=dropdown_options)
            log_x = st.sidebar.selectbox('Log axis on x', options=[True, False])
            log_y = st.sidebar.selectbox('Log axis on y', options=[True, False])
            #title = st.text_input(label='Title of chart')
            plot = px.violin(data_frame=df,x=x_values,
                             y=y_values,color=color_value,
                             hover_name=hover_name_value,
                             facet_row=facet_row_value,
                             facet_col=facet_column_value,box=box,
                             log_x=log_x, log_y=log_y,violinmode=violinmode,points=outliers,
                             template=template) #title=title

        except Exception as e:
            print(e)

    if chart_type == 'Box plots':
        st.sidebar.subheader('🔨plot Settings')

        try:
            x_values = st.sidebar.selectbox('X axis', index=length_of_options, options=dropdown_options)
            y_values = st.sidebar.selectbox('Y axis', index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options, options=dropdown_options)
            boxmode = st.sidebar.selectbox('Violin mode', options=['group', 'overlay'])
            outliers = st.sidebar.selectbox('Show outliers', options=[False, 'all', 'outliers', 'suspectedoutliers'])
            hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options, options=dropdown_options)
            facet_row_value = st.sidebar.selectbox("Facet row", index=length_of_options, options=dropdown_options, )
            facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
                                                      options=dropdown_options)
            log_x = st.sidebar.selectbox('Log axis on x', options=[True, False])
            log_y = st.sidebar.selectbox('Log axis on y', options=[True, False])
            notched = st.sidebar.selectbox('Notched', options=[True, False])
            #title = st.text_input(label='Title of chart')
            plot = px.box(data_frame=df, x=x_values,
                          y=y_values, color=color_value,
                          hover_name=hover_name_value,facet_row=facet_row_value,
                          facet_col=facet_column_value, notched=notched,
                          log_x=log_x, log_y=log_y, boxmode=boxmode, points=outliers,
                          template=template) #title=title

        except Exception as e:
            print(e)

    if chart_type == 'Sunburst':
        st.sidebar.subheader('🔨plot Settings')

        try:
            path_value = st.sidebar.multiselect(label='Path', options=dropdown_options)
            color_value = st.sidebar.selectbox(label='Color', options=dropdown_options)
            value = st.sidebar.selectbox("Value", index=length_of_options, options=dropdown_options)
            #title = st.text_input(label='Title of chart')

            plot = px.sunburst(data_frame=df,path=path_value,values=value,
                               color=color_value) #title=title 

        except Exception as e:
            print(e)

    if chart_type == 'Tree maps':
        st.sidebar.subheader('🔨plot Settings')

        try:
            path_value = st.sidebar.multiselect(label='Path', options=dropdown_options)
            color_value = st.sidebar.selectbox(label='Color', options=dropdown_options)
            value = st.sidebar.selectbox("Value", index=length_of_options, options=dropdown_options)
            #title = st.text_input(label='Title of chart')

            plot = px.treemap(data_frame=df,path=path_value,values=value,
                              color=color_value) # title=title 

        except Exception as e:
            print(e)

    if chart_type == 'Pie Charts':
        st.sidebar.subheader('🔨plot Settings')

        try:
            name_value = st.sidebar.selectbox(label='Name (Selected Column should be categorical)', options=dropdown_options)
            color_value = st.sidebar.selectbox(label='Color(Selected Column should be categorical)', options=dropdown_options)
            value = st.sidebar.selectbox("Value", index=length_of_options, options=dropdown_options)
            hole = st.sidebar.selectbox('Log axis on y', options=[True, False])
            #title = st.text_input(label='Title of chart')

            plot = px.pie(data_frame=df,names=name_value,hole=hole,
                          values=value,color=color_value) # title=title

        except Exception as e:
            print(e)

    if chart_type == 'Density contour':
        st.sidebar.subheader("🔨plot Settings")

        try:
            x_values = st.sidebar.selectbox('X axis', index=length_of_options,options=dropdown_options)
            y_values = st.sidebar.selectbox('Y axis',index=length_of_options, options=dropdown_options)
            z_value = st.sidebar.selectbox("Z axis", index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)
            hist_func = st.sidebar.selectbox('Histogram aggregation function', index=0,
                                             options=['count', 'sum', 'avg', 'min', 'max'])
            histnorm = st.sidebar.selectbox('Hist norm', options=[None, 'percent', 'probability', 'density',
                                                                  'probability density'], index=0)
            hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options,options=dropdown_options)
            facet_row_value = st.sidebar.selectbox("Facet row",index=length_of_options, options=dropdown_options,)
            facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
                                                      options=dropdown_options)
            marginalx = st.sidebar.selectbox("Marginal X", index=2,options=['rug', 'box', None,
                                                                         'violin', 'histogram'])
            marginaly = st.sidebar.selectbox("Marginal Y", index=2,options=['rug', 'box', None,
                                                                         'violin', 'histogram'])
            log_x = st.sidebar.selectbox('Log axis on x', options=[True, False],index=1)
            log_y = st.sidebar.selectbox('Log axis on y', options=[True, False], index=1)
            #title = st.text_input(label='Title of chart')
            plot = px.density_contour(data_frame=df,x=x_values,y=y_values, color=color_value,
                                      z=z_value, histfunc=hist_func,histnorm=histnorm,
                                      hover_name=hover_name_value,facet_row=facet_row_value,
                                      facet_col=facet_column_value,log_x=log_x,
                                      log_y=log_y,marginal_y=marginaly, marginal_x=marginalx,
                                      template=template) # title=title

        except Exception as e:
            print(e)

    if chart_type == 'Density heatmaps':
        st.sidebar.subheader("🔨plot Settings")

        try:
            x_values = st.sidebar.selectbox('X axis', index=length_of_options, options=dropdown_options)
            y_values = st.sidebar.selectbox('Y axis', index=length_of_options, options=dropdown_options)
            z_value = st.sidebar.selectbox("Z axis", index=length_of_options, options=dropdown_options)
            hist_func = st.sidebar.selectbox('Histogram aggregation function', index=0,
                                             options=['count', 'sum', 'avg', 'min', 'max'])
            histnorm = st.sidebar.selectbox('Hist norm', options=[None, 'percent', 'probability', 'density',
                                                                  'probability density'], index=0)
            hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options, options=dropdown_options)
            facet_row_value = st.sidebar.selectbox("Facet row", index=length_of_options, options=dropdown_options, )
            facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
                                                      options=dropdown_options)
            marginalx = st.sidebar.selectbox("Marginal X", index=2, options=['rug', 'box', None,
                                                                             'violin', 'histogram'])
            marginaly = st.sidebar.selectbox("Marginal Y", index=2, options=['rug', 'box', None,
                                                                             'violin', 'histogram'])
            log_x = st.sidebar.selectbox('Log axis on x', options=[True, False], index=1)
            log_y = st.sidebar.selectbox('Log axis on y', options=[True, False], index=1)
            #title = st.text_input(label='Title of chart')
            plot = px.density_heatmap(data_frame=df, x=x_values, y=y_values,
                                      z=z_value, histfunc=hist_func, histnorm=histnorm,
                                      hover_name=hover_name_value, facet_row=facet_row_value,
                                      facet_col=facet_column_value, log_x=log_x,
                                      log_y=log_y, marginal_y=marginaly, marginal_x=marginalx,
                                      template=template) #  title=title

        except Exception as e:
            print(e)

    st.subheader("Chart")
    st.info('Use the sidebar for settings', icon="ℹ️")
    st.plotly_chart(plot)
    st.subheader("")
    show_export_format(plot)







@st.cache
def load_dataframe(upload_file):
    try:
        df = pd.read_csv(upload_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(upload_file)

    columns = list(df.columns)
    columns.append(None)

    return df, columns



if upload_file is not None:
    try:
        df, columns = load_dataframe(upload_file=upload_file)
            
    
        
        st.sidebar.subheader("🔧Basic Settings")

        theme_selection = st.sidebar.selectbox(label="Themes",
                                                options=['plotly', 'plotly_white',
                                                        'ggplot2',
                                                        'seaborn', 'simple_white'])
        
        chart_type = st.sidebar.selectbox(label="Chart Type",
                                            options=['Scatter plots', 'Density contour',
                                                    'Sunburst','Pie Charts','Density heatmaps',
                                                    'Histogram', 'Box plots','Tree maps',
                                                    'Violin plots', ])  # 'Line plots',

        graph_controls(chart_type=chart_type, df=df, dropdown_options=columns, template=theme_selection)
    except Exception as e:
            print(e)





