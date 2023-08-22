# %%
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# %%
plt.style.use('default')

st.set_page_config(
    page_title = 'Hello',
    page_icon = '🌾',
    layout = 'wide'
)

# %%
st.sidebar.success("Select a model for interpretation above.")

# %%
st.write("# Welcome to Interpretable Structure-Activity Relationship (ISAR) for seed nanopriming! 👋")

st.markdown(
    """
    This is online local interpretation web app for this article [citation] 🌱

    **👈 Select a model for interpretation from the sidebar** to see local interpretation based on different models

    ### Want to learn more interpretation results 👀
    - See our [paper](https://pubs.rsc.org/en/content/articlelanding/2023/nr/d3nr02322b)
    - Check out [source code for interpretation](https://github.com/YuHengjie/ISAR_Seed_Nanopriming_Code)
    
    ### Some notes 🗈
    - Model interpretation cannot distinguish correlation from causation
    - We prefer to use SHAP for local interpretation as its results are more consistent with the model's predicted probabilities (that is what we do in the published article)
    - Self-interpretable machine learning, interpretable on the construction of the model, may be a better choice if it can achieve reasonable predictive accuracy

    ### See more ISAR articles based on model interpretation 🔍
    - Integrating machine learning interpretation methods for investigating [nanoparticle uptake during seed priming and its biological effects](https://pubs.rsc.org/en/content/articlelanding/2022/NR/D2NR01904C)
    - Interpretable machine learning for investigating [complex nanomaterial-plant-soil interactions](https://pubs.rsc.org/en/content/articlelanding/2022/en/d2en00181k)
    - Predicting and investigating [cytotoxicity of nanoparticles by translucent machine learning](https://www.sciencedirect.com/science/article/pii/S0045653521006330?via%3Dihub)
    - Averaging strategy for interpretable machine learning on small datasets to [understand element uptake after seed nanotreatment](https://pubs.acs.org/doi/abs/10.1021/acs.est.3c01878)
    
    ### Source code of this web app is available ✈️
    - [Source code](https://github.com/YuHengjie/ISAR_Seed_Nanopriming_Web) developed by 🐘 Hengjie Yu

    ### Citation 👏🏽
    - If you found this web app useful for your blog post, research article or product, I would be grateful if you would cite this article [Nanoscale 2023, DOI: 10.1039/d3nr02322b](https://pubs.rsc.org/en/content/articlelanding/2023/nr/d3nr02322b)

"""
)
