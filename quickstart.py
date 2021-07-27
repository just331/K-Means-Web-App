from scipy.sparse import data
import streamlit as st
import pandas as pd
from seaborn import color_palette
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style()


# Helper Functions

@st.cache
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv")
    return df


df = load_data()


def run_kmeans(df, n_clusters=2):
    kmeans = KMeans(n_clusters, random_state=0).fit(df[["Age", "Income"]])

    fig, ax = plt.subplots(figsize=(16, 19))

    ax = sns.scatterplot(
        ax=ax,
        x=df.Age,
        y=df.Income,
        hue=kmeans.labels_,
        palette=color_palette("colorblind", n_colors=n_clusters),
        legend=None,
    )

    for ix, [age, income] in enumerate(kmeans.cluster_centers_):
        ax.scatter(age, income, s=200, c="#a8323e")
        ax.annotate(
            f"Cluster #{ix + 1}",
            (age, income),
            fontsize=15,
            color="#a8323e",
            xytext=(age + 5, income + 3),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a8323e", lw=2),
            ha="center",
            va="center",
        )

    return fig


# Sidebar
sidebar = st.sidebar
df_display = sidebar.checkbox("Display Raw Data", value=True)

n_clusters = sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
)


# Main
st.title("Interactive K-Means Clustering")

st.write("Dataset used in this analysis")

st.write(run_kmeans(df, n_clusters=n_clusters))

if df_display:
    st.write(df)
