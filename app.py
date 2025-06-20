import streamlit as st
import seaborn as sea
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# Initialize session state for the uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
    st.session_state.df = None
    st.session_state.cleaned_df = None

# Function to handle file upload
def handle_file_upload():
    file = st.file_uploader("Upload your dataset, nerd.", type=["csv", "json", "xls", "xlsx"])
    if file is not None:
        st.session_state.uploaded_file = file
        file_extension = file.name.split(".")[-1]
        
        if file_extension == "csv":
            df = pd.read_csv(file)
        elif file_extension == "json":
            df = pd.read_json(file)
        elif file_extension in ["xls", "xlsx"]:
            excel_data = pd.ExcelFile(file)
            sheet_names = excel_data.sheet_names
            sheet_choice = st.selectbox("Select a sheet", sheet_names)

            if sheet_choice:
                df = pd.read_excel(file, sheet_name=sheet_choice)
        
        st.write("Here is your data:")
        st.dataframe(df)
        st.session_state.df = df.copy()
        return df

def data_exploration():
    if st.session_state.df is not None:
        col1, col2, col3 = st.columns([1,1,1])
    
        with col1:
            with st.popover("Show Head", use_container_width=True):
                st.dataframe(st.session_state.df.head())
            with st.popover("Column Names", use_container_width=True):
                st.dataframe(st.session_state.df.columns)
        with col2:
            with st.popover("Show Tail", use_container_width=True):
                st.dataframe(st.session_state.df.tail())
            with st.popover("Unique Values", use_container_width=True):
                st.dataframe(st.session_state.df.nunique())
                if st.checkbox("View unique values"):
                    for column in st.session_state.df.columns:
                        st.write(f"Unique values for {column}:")
                        st.write(st.session_state.df[column].unique())
        with col3:
            with st.popover("Describe", use_container_width=True):
                st.dataframe(st.session_state.df.describe())
            if st.session_state.cleaned_df is not None:
                with st.popover("Data Type", use_container_width=True):
                    st.dataframe(st.session_state.cleaned_df.dtypes)
            else:
                with st.popover("Data Type", use_container_width=True):
                    st.dataframe(st.session_state.df.dtypes)
            


def data_cleaning():
    if st.session_state.df is not None:
        df = st.session_state.df
        cleaned_df = df.copy()
        
        if st.sidebar.checkbox('Remove Duplicate Values'):
            cleaned_df = cleaned_df.drop_duplicates()
            st.title("Dataframe with Duplicated values Removed:")
            st.dataframe(cleaned_df)
        
        if st.sidebar.checkbox('Remove NULL Values'):
            na_handling=st.sidebar.selectbox("Filter NULL Values by:",["Select Method", "Removing Rows", "Filling NaNs"])
            if na_handling == "Removing Rows":
                if st.session_state.df is not None:
                    cleaned_df = cleaned_df.dropna().reset_index(drop=True)
                    
            if na_handling == "Filling NaNs":
                if st.session_state.df is not None:
                    method = st.sidebar.selectbox("Select the Method to fill:",['Select Parameter', 'Mean', 'Median', 'Mode', 'Standard Deviation', 'Custom Value'])
                    if method == 'Mean':
                        columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns
                        for column in columns:
                            mean = int(cleaned_df[column].mean())
                            cleaned_df[column] = cleaned_df[column].fillna(mean)
                    if method == 'Median':
                        columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns
                        for column in columns:
                            median = int(cleaned_df[column].median())
                            cleaned_df[column] = cleaned_df[column].fillna(median)
                        
                        
                    if method == 'Mode':
                        columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns
                        for column in columns:
                            mode_series = cleaned_df[column].mode()
                            if not mode_series.empty:
                                mode = int(mode_series[0])  # Extract the first value from the mode series and convert to integer
                                cleaned_df[column] = cleaned_df[column].fillna(mode)
                        

                    if method == 'Standard Deviation':
                        columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns
                        for column in columns:
                            standard_deviation = int(cleaned_df[column].std())
                            cleaned_df[column] = cleaned_df[column].fillna(standard_deviation)
                        
                        
                    if method == 'Custom Value':
                        columns = list(df.columns)
                        col1, col2, col3 = st.columns(3)
                        for i, column in enumerate(columns):
                            with (col1, col2, col3)[i % 3]:
                                default_value = st.text_input(f"Default value for {column}", key=column)
                                if default_value:  # Only fill NaNs if a default value is provided
                                    cleaned_df[column].fillna(default_value, inplace=True)
                        

        
        # st.session_state.cleaned_df = cleaned_df    
        if st.sidebar.checkbox('Remove Garbage Values'):
            garbage_handling = st.sidebar.selectbox("Handle Garbage Value By:", ['Select an option', 'Remove Rows', 'Custom Value'])
            
            if garbage_handling == 'Remove Rows':
                columns = cleaned_df.select_dtypes(exclude = ['float64', 'int64']).columns
                for column in columns:
                    cleaned_df = cleaned_df.dropna().reset_index(drop=True)

                    
            if garbage_handling == 'Custom Value':
                columns = list(cleaned_df.select_dtypes(exclude = ['float64', 'int64']))
                col1, col2, col3 = st.columns(3)
                for i, column in enumerate(columns):
                    with (col1, col2, col3)[i % 3]:
                        default_value = st.text_input(f"Default value for {column}", key=column)
                        if default_value:  # Only fill NaNs if a default value is provided
                            cleaned_df[column].fillna(default_value, inplace=True)
                st.session_state.cleaned_df = cleaned_df
            object_columns = cleaned_df.select_dtypes(include=['object']).columns
            cleaned_df[object_columns] = cleaned_df[object_columns].astype('string')
            
            st.session_state.cleaned_df = cleaned_df
            st.write(cleaned_df)
        
def remove_outliers(df):
    # Select only numeric columns
    columns = list(df.select_dtypes(include=['float64', 'int64']).columns)
    
    # Initialize sets to store indices
    all_indices = set(df.index)
    outlier_indices = set()
    
    # Loop through each numeric column
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Get indices of outlier values
        outliers = set(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index)
        
        # Update the set of outlier indices
        outlier_indices |= outliers  # Union of outlier indices across all columns

    # Normal indices are those not in outlier_indices
    normal_indices = all_indices - outlier_indices
    
    # Convert sets to lists for indexing
    normal_indices = list(normal_indices)
    outlier_indices = list(outlier_indices)
    
    # Create DataFrames for cleaned data and outliers
    cleaned_df = df.loc[normal_indices]
    dirty_df = df.loc[outlier_indices]
    
    return cleaned_df, dirty_df


    
    
def streamlit_menu():
    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Visualization", "Processing", "Training"],  # required
        icons=[" ", " ", " ", " "],   # Set icons to empty strings to remove icons
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={"container": {
                "border-radius": "25px",
                },
                "nav-link": {
                "border-radius": "25px",
                "--hover-color": "#000000",
            }}
    )
    return selected

selected = streamlit_menu()

def Home():
    st.title("Welcome to the Data Visualization and Outlier Detection App")
    
    st.markdown("""
    This project is designed to help you visualize datasets and detect outliers using various methods. 
    It includes functionalities to:
    
    - *Upload Datasets*: Upload your CSV files and view them in an interactive table.
    - *Visualize Data*: Create various types of plots (scatter, bar, line, pie, histogram) using Plotly.
    - *Remove Outliers*: Apply the IQR method to remove outliers from your dataset.
    - Prediction: Train a model and make predictions on your dataset.
    """)
    
            
def Visualization():
    st.title(f"You have selected {selected}")
    handle_file_upload()
    if st.session_state.df is not None:
        df = st.session_state.df
        file = st.session_state.uploaded_file

        st.sidebar.title("Filter Data")
        selected_columns = st.sidebar.multiselect("Select columns to filter", df.columns)
        filtered_df = df.copy()
        
        if selected_columns:
            for col in selected_columns:
                unique_values = st.sidebar.multiselect(f"Unique values for {col}", df[col].unique())
                if unique_values:
                    filtered_df = filtered_df[filtered_df[col].isin(unique_values)]
                        
            st.write("Filtered data:")
            st.dataframe(filtered_df)
                
        st.sidebar.title("Plot Data")
        plot_types = ["Please select an option", "Scatter Plot", "Bar Graph", "Line Graph", "Pie Chart", "Histogram", "Correlation Heatmap", "Pairplot"]

# Create a selectbox with the placeholder as the default option
        plot_type = st.sidebar.selectbox("Choose plot type", plot_types)
        
        if plot_type in ["Scatter Plot", "Bar Graph", "Line Graph"]:
            x_axis = st.sidebar.selectbox("X-Axis", options=filtered_df.columns)
            y_axis = st.sidebar.selectbox("Y-Axis", options=filtered_df.columns)

            if plot_type == "Scatter Plot":
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis)
                fig.update_traces(marker=dict(color='rgba(135, 206, 235, 0.8)', line=dict(color='rgba(0, 0, 139, 1.0)', width=1)))
                fig.update_layout(
                    title="Scatter Plot",
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    # xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                    # yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
                )
            elif plot_type == "Bar Graph":
                fig = px.bar(filtered_df, x=x_axis, y=y_axis)
                fig.update_traces(width=0.5, marker_line_width=1.5, marker_line_color='rgb(8,48,107)')
                fig.update_layout(
                    title="Bar Graph",
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    # xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                    # yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
                )
            elif plot_type == "Line Graph":
                fig = px.line(filtered_df, x=x_axis, y=y_axis)
                fig.update_traces(line=dict(width=2, dash='dash'), marker=dict(size=10, symbol='circle'))
                fig.update_layout(
                    title="Line Graph",
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    # xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                    # yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
                )
                
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        elif plot_type == "Histogram":
            x_axis = st.sidebar.selectbox("X-Axis", options=filtered_df.columns)
            bins = st.sidebar.slider("Number of bins", min_value=5, max_value=100, value=30)
            fig = px.histogram(filtered_df, x=x_axis, nbins=bins)
            fig.update_layout(
                title="Histogram",
                xaxis_title=x_axis,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                # xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                # yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
            )
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.plotly_chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        elif plot_type == "Pie Chart":
            x_axis = st.sidebar.selectbox("Values", options=filtered_df.columns)
            if filtered_df[x_axis].nunique() <= 10:
                fig = px.pie(filtered_df, names=x_axis)
                fig.update_traces(
                    pull=[0.1 if val == max(filtered_df[x_axis]) else 0 for val in filtered_df[x_axis]],
                    textinfo='percent+label'
                )
                fig.update_layout(
                    title="Pie Chart",
                    plot_bgcolor='rgba(0, 0, 0, 0)'
                )
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Too many unique values for pie chart. Select a column with fewer unique values.")


        elif plot_type == "Correlation Heatmap":
                corr_input = df.select_dtypes(include=['float', 'int'])
                corr = corr_input.corr()
                # fig, ax = plt.subplots(figsize=(10, 8))
                # sea.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='Set1', ax=ax)
                # plt.title("Heatmap of Correlation Matrix")
                # ax.set_facecolor('none')

                # # Change the background color of the figure to transparent
                # plt.gcf().patch.set_facecolor('none')

                # # Customize text colors
                # ax.title.set_color('white')  # Title color
                # ax.xaxis.label.set_color('white')  # X-axis label color
                # ax.yaxis.label.set_color('white')  # Y-axis label color
                # ax.tick_params(axis='x', colors='white')  # X-axis ticks color
                # ax.tick_params(axis='y', colors='white')  # Y-axis ticks color

                # # Add a title
                # plt.title('Correlation Heatmap', color='white')

                # # Save the figure with a transparent background
                # plt.savefig('heatmap.png', transparent=True)
                
                # # Display the heatmap in Streamlit
                # st.pyplot(fig)
                
                fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis',
                colorbar=dict(title='Correlation')
                ))
                fig.update_layout(
                    title='Correlation Heatmap',
                    xaxis_title='Features',
                    yaxis_title='Features',
                    xaxis=dict(tickvals=list(corr.columns), ticktext=list(corr.columns)),
                    yaxis=dict(tickvals=list(corr.columns), ticktext=list(corr.columns)),
                )
                
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.plotly_chart(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
        elif plot_type == "Pairplot":
            # Assuming 'pair' is your DataFrame with float and int types
            pair = df.select_dtypes(include=['float', 'int'])
            
            pair = st.sidebar.multiselect("Choose Columns", options=pair.columns)
            
            pair = df[pair]
 
            # Create a PairGrid using seaborn's pairplot
            g = sea.pairplot(pair)

            # Customize appearance using matplotlib
            # Iterate over each axis in the PairGrid to adjust properties
            for ax in g.axes.flatten():
                ax.set_facecolor('none')  # Set background color of individual subplots
                ax.title.set_color('white')  # Title color
                ax.xaxis.label.set_color('white')  # X-axis label color
                ax.yaxis.label.set_color('white')  # Y-axis label color
                ax.tick_params(axis='x', colors='white')  # X-axis ticks color
                ax.tick_params(axis='y', colors='white')  # Y-axis ticks color

            # Set the background color of the figure to transparent
            plt.gcf().patch.set_facecolor('none')



            # Save the figure with a transparent background
            plt.savefig('heatmap.png', transparent=True)

            # Display the pairplot in Streamlit
            st.pyplot(g.figure)
        
            
                
                
                
        
def Processing():
    st.title(f"You have selected {selected}")
    handle_file_upload()
    
    if st.session_state. uploaded_file is not None:
        choice=st.sidebar.multiselect('Select an option:',options=["Exploratory Data Analysis", "Data Cleaning", "Data Preprocessing"])
        if "Exploratory Data Analysis" in choice:
            st.subheader("Data Exploration:")
            data_exploration()
        if "Data Cleaning" in choice:
            st.subheader("Data Cleaning:")
            data_cleaning()
        if "Data Preprocessing" in choice:
            st.subheader("Data Preprocessing")
            return()
    
    
def Training():
    st.title(f"You have selected {selected}")
    handle_file_upload()
    
if selected == "Home":
    Home()
elif selected == "Visualization":
    Visualization()
elif selected == "Processing":
    Processing()
elif selected == "Training":
    Training()