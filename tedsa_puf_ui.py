import streamlit as st
import pandas as pd
import pyxlsb
import numpy as np
import seaborn as sns      #package for data visulaizations
import plotly.express as px #package for interactive data visualizations
import matplotlib.pyplot as plt #package for data visulaizations
st.set_page_config(layout="wide")
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as stc



def add_percent(num):
    num=num*100
    num=round(num,3)
    return str(num)+" %"




def correlations(df):
    correlation = df.corr()
    filtered_columns = correlation.loc[correlation["NOPRIOR"] > 0.05,'NOPRIOR']
    cols_to_use_in_model = correlation.index
    # correlation matrix of the whole dataframe witt all variables
    fig,ax = plt.subplots(figsize=(30, 25))
    st.subheader("Correlation Matrix of the whole Dataframe witt All Variables")
    ax=sns.heatmap(df.corr(), annot = True, fmt= '.2f')
    st.pyplot(fig)
    # Making a new dataset with columns to use for the model
    df_filtered = df[correlation.index]
    st.subheader("Making Correlation Matrix for new Filtered Dataset")
    fig,ax = plt.subplots(figsize=(30, 25))
    ax=sns.heatmap(df_filtered.corr(), annot = True, fmt= '.2f')
    st.pyplot(fig)
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(df_filtered)
    # Getting variance for each feature
    variance = pca.explained_variance_ratio_
    # making a dataframe 
    pca_compoenents = ["PCA1","PCA2","PCA3","PCA4","PCA5"]
    pca_df = pd.DataFrame({"PCA_components":df_filtered.columns,"variance ratio":variance})
    st.subheader("Comappring Among FeaturesConsidering the Contribution of each Principal Component")
    fig,ax = plt.subplots(figsize=(30, 5))
    ax=sns.barplot(pca_df["PCA_components"],pca_df["variance ratio"])
    plt.xticks(rotation=90)
    st.pyplot(fig)
    st.subheader("Comappring among Features Considering the Contribution of each Principal Component (Log Scale)")
    fig,ax = plt.subplots(figsize=(30, 5))
    ax=sns.barplot(pca_df["PCA_components"],pca_df["variance ratio"])
    plt.xticks(rotation=90)
    plt.title("Variance ratio of variables with log scale on y-axis")
    plt.yscale("log")
    st.pyplot(fig)
    feat_list={"CASEID":"Case Identification Number",
    "ADMYR": "Year of Admission",
    "AGE": "Age at Admission",
    "GENDER": "Gender",
    "RACE": "Race",
    "ETHNIC": "Ethnicity",
    "MARSTAT": "Marital Status",
    "EDUC": "Education",
    "EMPLOY": "Employment Status",
    "DETNLF": "Detailed not in Labor Force",
    "PREG": "Pregnant at Admission",
    "VET": "Veteran Status",
    "LIVARAG": "Living Arrangements",
    "PRIMINC": "Source of Income/Support",
    "ARRESTS": "Arrests in past 30 days",
    "STFIPS": "Census State FIPS Code",
    "CBSA2010": "CBSA 2010 Code",
    "REGION": "Census Region",
    "DIVISION": "Census Division",
    "SERVICES": "Type of Treatment Service/Setting",
    "METHUSE": "Medication-assisted Opioid Therapy",
    "DAYWAIT": "Days Waiting to Enter Substance use Treatment",
    "PSOURCE": "Referral Source",
    "DETCRIM": "Detailed Criminal Justice Referral",
    "NOPRIOR": "Previous Substance use Treatment Episodes",
    "SUB1": "Substance Use (Primary)",
    "ROUTE1": "Route of Administration (Primary)",
    "FREQ1": "Frequency of Use (Primary)",
    "FRSTUSE1": "Age at First Use (primary)",
    "SUB2": "Substance Use (Secondary)",
    "ROUTE2": "Route of Administration (Secondary)",
    "FREQ2": "Frequency of Use (Secondary)",
    "FRSTUSE2": "Age at First Use (Secondary)",
    "SUB3": "Substance Use (Tertiary)",
    "ROUTE3": "Route of Administration (Tertiary)",
    "FREQ3": "Frequency of Use (Tertiary)",
    "FRSTUSE3": "Age at First Use (Tertiary)",
    "IDU": "Current IV Drug Use Reported at Admission",
    "ALCFLG": "Alcohol Reported at Admission",
    "COKEFLG": "Cocaine/Crack Reported at Admission",
    "MARFLG": "Marijuana/Hashish Reported at Admission",
    "HERFLG": "Heroin Reported at Admission",
    "METHFLG": "Non-rx Methadone Reported at Admission",
    "OPSYNFLG": "Other Opiates/Synthetics Reported at Admission",
    "PCPFLG": "PCP reported at Admission",
    "HALLFLG": "Hallucinogens Reported at Admission",
    "MTHAMFLG": "Methamphetamine/speed Reported at Admission",
    "AMPHFLG": "Other Amphetamines Reported at Admission",
    "STIMFLG": "Other Stimulants Reported at Admission",
    "BENZFLG": "Benzodiazepines Reported at Admission",
    "TRNQFLG": "Other Tranquilizers Reported at Admission",
    "BARBFLG": "Barbiturates Reported at Admission",
    "SEDHPFLG": "Other Sedatives/Hypnotics Reported at Admission",
    "INHFLG": "Inhalants Reported at Admission",
    "OTCFLG": "Over-the-Counter Medication Reported at Admission",
    "OTHERFLG": "Other Drug Reported at Admission",
    "ALCDRUG": "Substance Use Type",
    "DSMCRIT": "DSM Diagnosis (SuDS 4 or SuDS 19)",
    "PSYPROB": "Co-occurring Mental and Substance Use Disorders",
    "HLTHINS": "Health Insurance",
    "PRIMPAY": "Payment Source, Primary (Expected or Actual)",
    "FREQ_ATND_SELF_HELP": "Attendance at Substance Use Self-Help Groups in Past 30 Days"}
    df_temp=pca_df
    # df_temp=df_temp['variance ratio']*100
    df_temp['variance ratio']=df_temp['variance ratio'].apply(lambda x:add_percent(x))
    for f in feat_list.keys():
        for dex,i in enumerate(df_temp["PCA_components"]):
            if i==f:
                df_temp.at[dex,'PCA_components']=feat_list[f]
    df_temp.rename(columns = {"PCA_components": "Features","variance ratio": "Importance"}, inplace = True)
    st.subheader("Featres Rankings by Importance")
    st.write(df_temp)
    st.snow()


def make_prediction(input_array):
    input_array = np.array(input_array)
    input_2darray=input_array.reshape(1, 7)
    scaler = StandardScaler() # Standard Scaler
    scaler.fit(input_2darray)  # Train Standard Scaler
    scaled_data= scaler.transform(input_2darray)
    res=classifier.predict(scaled_data)
    return res[0]




# Function to perform all EDA
def perform_eda(df, name=""):
    # Printing basic detail of data like name, size, shape
    st.write(f"EDA of {str(name)} Data....")
    st.write(f"Size {df.size}")
    st.write(f"Columns {df.shape[1]}")
    st.write(f"Records {df.shape[0]}")
    st.write("="*50)
    
    # # Printing top 5 records of data
    st.write("First Look of Data....")
    st.write(df.head(10))
    
    
def make_plots(feature, title="", limited=False, n=10):
    print("Total unique values are: ", len(feature.value_counts()), "\n\n")
    print("Category\tValue\n")
    if limited:
        data = feature.value_counts()[0:n]
    else:
        data = feature.value_counts()
    print(data)
    categories_num = len(data)
    #plotting bar-plot and pie chart
    sns.set_style('darkgrid')
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45)
    fig1,ax=plt.subplots()
    plot = sns.barplot(x=data.index, y=data.values, edgecolor="white", palette=sns.palettes.color_palette("icefire"))
    total = len(feature)
    for p in plot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.08
        y = p.get_y() + p.get_height()
        plot.annotate(percentage, (x, y), size = 12)
    col1,col2=st.columns(2)
    with col1:
    	st.pyplot(fig1)


    fig,ax=plt.subplots()

    labels = data.index
    plt.pie(x=data, autopct="%.1f%%", explode=[0.02]*categories_num, labels=labels, pctdistance=0.5)
    plt.title(title, fontsize=16)
    with col2:
	    st.pyplot(fig)

st.write("Welcome To")
custom_banner="""<div style="font-size=60px;font-weight:bolder,background-color:#fff;padding:10px;border-radius:10px;border:5px solid #464e5f;text-align:center;">
<span style="color:green;font-size:70px">TEDSA PUF DATA ANALYSIS</span>
</div>
"""
stc.html(custom_banner)

# st.title("TEDSA PUF")


uploaded_file=st.file_uploader("Import Data File",type=['xlsb'],accept_multiple_files=False)
if uploaded_file is not None:
    df=pd.read_excel(uploaded_file)

    # correlations(df)
    st.subheader("Over View of the Data")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.write("Shape of the Data")
        st.subheader(df.shape)
    with col2:
        st.write("Numerical Features")
        num_cols = df.select_dtypes(np.number).columns
        st.subheader(len(num_cols))
    with col3:
        st.write("Catgorical Features")
        cat_cols = df.select_dtypes(np.object).columns
        st.subheader(len(cat_cols))
    with col4:
        st.write("Size of the Data")
        st.subheader(df.size)
    st.dataframe(df.head(10))
    num_cols=df.select_dtypes(np.number).columns
    selected_feature=st.selectbox('Select Feature',num_cols)
    make_plots(df[selected_feature])
    st.subheader("Statistical Properties of the Data")
    st.dataframe(df.describe())
    correlations(df)





    # with st.expander("Model"):
    #     col1,col2,col3,col4=st.columns(4)
    #     with col1:
    #         PSYPROB=st.text_input("Enter PSYPROB")
    #     with col2:
    #         SUB1=st.text_input("Enter SUB1")
    #     with col3:
    #         SUB2=st.text_input("Enter SUB2")
    #     with col4:
    #         SUB3=st.text_input("Enter SUB3")    
    #     col1,col2,col3=st.columns(3)
    #     with col1:
    #         FREQ1=st.text_input("Enter FREQ1")
    #     with col2:
    #         FREQ2=st.text_input("Enter FREQ2")
    #     with col3:
    #         FREQ3=st.text_input("Enter FREQ3")
    #     inputs=[PSYPROB,SUB1,SUB2,SUB3,FREQ1,FREQ2,FREQ3]
    #     if st.button("Predict"):
    #         st.title("Predicted Results")
    #         st.snow()
    #         st.subheader(make_prediction(inputs))



