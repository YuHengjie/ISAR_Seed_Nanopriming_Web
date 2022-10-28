# %%# %%
from PIL import Image
import streamlit as st
import numpy as np 
import pandas as pd 

import joblib
import lightgbm
from sklearn.tree import DecisionTreeClassifier
import shap

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import lime
import lime.lime_tabular

# %%
plt.style.use('default')

st.set_page_config(
    page_title = 'Decision Tree',
    page_icon = '🌴',
    layout = 'wide'
)

# %%
st.markdown("<h1 style='text-align: center; color: black;'>Local interpretation based on the Decision Tree model </h1>", unsafe_allow_html=True)

# %%
# understand the dataset
df_raw = pd.read_excel('data/dataset_DecisionTree.xlsx',index_col = 0)
X = df_raw.drop(columns=['Root dry weight']) # 'TEM size (nm)'
X_raw = X.copy() # 用于shap可视化

# %%
X_test_index = np.load('data/X_test_index_RDW.npy',allow_pickle=True)
X_cv_index = np.load('data/X_cv_index_RDW.npy',allow_pickle=True)
X_cv = X.loc[X_cv_index]

# %%
# side-bar 
def user_input_features(dataset=X_raw,test_index=X_test_index):
    st.sidebar.header('Select or input an instance')
    st.sidebar.write(f'Select an instance from dataset for local interpretation: 0~{len(df_raw)-1}')
    index_list = ['None']
    for i in range(0,len(df_raw),1):
        index_list.append(i)
    index = st.sidebar.selectbox('Instance index', index_list)

    st.sidebar.write('Or input parameters below (set the instance index to None) ⬇️')
    a1 = st.sidebar.slider(X.columns[0], float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), 100.0)
    a2 = st.sidebar.slider(X.columns[1], X.iloc[:,1].min(), X.iloc[:,1].max(), 60.0)
    a3 = st.sidebar.slider(X.columns[2], X.iloc[:,2].min(), X.iloc[:,2].max(), 0.0)
    customize_instance = [a1,a2,a3]

    if index=='None':
        instance_type = 'custom'
        output = customize_instance
    else:
        if index in test_index:
            instance_type = 'test set'
        else:
            instance_type = 'train set'
        output = dataset.iloc[index,:]

    return instance_type, output

instance_type, outputdf = user_input_features()

# %%
st.title('Dataset')
if st.button('View some random data'):
    st.write(df_raw.sample(5))

# %%
st.write('Decision Tree is employed to make predictions. Train/test ratio: 3/1. Dataset split approach: stratified shuffle split.')
st.write(f'Dataset size: {len(df_raw)}. 0 means low level in root dry weight, 1 means high level in root dry weight. It is a balanced dataset.')

# %%
model_input = outputdf.copy()
model_input = pd.DataFrame([model_input], columns= X.columns)

# %%
model = joblib.load('model/dtmodel.pkl')
predict_proba_local = model.predict_proba(model_input)[0]
predict_proba_local = predict_proba_local[::-1]

# %%
outputdf = pd.DataFrame([outputdf], columns= X.columns)
st.title('Make prediction in real time')

placeholder1 = st.empty()
with placeholder1.container():
    f1,f2 = st.columns([4,1])
    with f1:
        st.write(f'This is a {instance_type} instance.')
        st.write(outputdf)
    with f2:
        fig, ax= plt.subplots(figsize = (2,1))
        bars = ['High','Low']
        plt.style.use('seaborn-ticks')
        plt.margins(0.05)
        plt.barh(range(0,len(predict_proba_local)), predict_proba_local,color=['#FF0050','#008BFA'])

        for i in range(0,len(bars)):
            if predict_proba_local[i]<0.25:
                if i==1:
                    plt.text(predict_proba_local[i]+0.12,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='#008BFA',)
                else:
                    plt.text(predict_proba_local[i]+0.12,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='#FF0050',)

            else:
                plt.text(predict_proba_local[i]/2,i-0.15,"%.2f" %predict_proba_local[i],ha = 'center',color='w',)


        ax.set_xticklabels([])

        ax1=plt.gca()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)

        ax1.tick_params(top=False,
                    bottom=False,
                    left=False,
                    right=False)

        plt.yticks(range(0,len(predict_proba_local)), bars)
        plt.title('Prediction probabilities')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

# %%
explainer = shap.TreeExplainer(model=model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(model_input)[1][0]

# %%
st.title('Prediction-level interpretation in real time')
placeholder2 = st.empty()
with placeholder2.container():
    f1,f2 = st.columns(2)
    with f1:
        class ShapObject:
            
            def __init__(self, base_values, data, values, feature_names):
                self.base_values = base_values # Single value
                self.data = data # Raw feature values for 1 row of data
                self.values = values # SHAP values for the same row of data
                self.feature_names = feature_names # Column names
        st.markdown("<h6 style='text-align: center; color: gray;'>SHAP</h6>", unsafe_allow_html=True)      
        shap_object = ShapObject(base_values = explainer.expected_value[1],
                                values = shap_values,
                                feature_names = X.columns,
                                data = outputdf.iloc[0,:])

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.waterfall_plot(shap_object, show=False)
        st.pyplot(bbox_inches='tight')

    with f2:
        st.markdown("<h6 style='text-align: center; color: gray;'>LIME</h6>", unsafe_allow_html=True)      

        explainer_lime = lime.lime_tabular.LimeTabularExplainer(np.array(X_cv.values), 
                                                        feature_names=X_cv.columns,
                                                        discretize_continuous=True,random_state=42)
        exp = explainer_lime.explain_instance(np.array(model_input.values[0]), model.predict_proba,
                                          num_features=8)
        lime_list = exp.as_list()

        bars = []
        height = []
        for i in range(0,len(lime_list)):
            bars.append(lime_list[i][0])
            height.append(lime_list[i][1])
        bars = bars[::-1]
        height = height[::-1]

        colors = ['#008BFA']
        for i in range(1,len(lime_list)):
            colors.append('#008BFA')

        for i in range(0,len(lime_list)):
            if height[i] >= 0:
                colors[i] = '#FF0050'

        fig, ax= plt.subplots(figsize = (6,2.8))
        plt.style.use('seaborn-ticks')
        plt.margins(0.05)
        plt.grid(linestyle=(0, (1, 6.5)),color='#B0B0B0',zorder=0)
        plt.barh(range(0,len(lime_list)), height,color=colors,edgecolor = "none", zorder=3)
        plt.yticks(range(0,len(lime_list)), bars,)

        ax1=plt.gca()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(True)
        ax1.spines['left'].set_visible(False)

        ax1.tick_params(top=False,
                    bottom=True,
                    left=False,
                    right=False)

        for i in range(0,len(bars)):
            if abs(height[i])<=0.013:
                if height[i]>0:
                    plt.text(height[i]+0.01,i-0.1,"%.2f" %height[i],ha = 'center',color='#FF0050',)
                else:
                    plt.text(height[i]-0.01,i-0.1,"%.2f" %height[i],ha = 'center',color='#008BFA',)
            else:
                plt.text(height[i]/2,i-0.1,"%.2f" %height[i],ha = 'center',color='w',)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
st.write('The custome instances may by valid points since they ignore feature distribution and feature correlation.')
st.write('SHAP is used for local interpretation in the published article. LIME is employed to compare with SHAP only in this web app.')
