import os
import pandas as pd
import streamlit as st
import plotly.express as px

MASTER_PATH  = os.path.expanduser("~/absa_project/output/master_triplets.csv")
SUMMARY_PATH = os.path.expanduser("~/absa_project/output/summary_stats.csv")

COLORS = {"Positive": "#2E7D32", "Negative": "#C62828", "Neutral": "#607D8B"}

st.set_page_config(page_title="ABSA Dashboard", layout="wide")
st.title("Aspect-Based Sentiment Analysis — Amazon Electronics")

@st.cache_data
def load_data():
    master  = pd.read_csv(MASTER_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    return master, summary

master, summary = load_data()

# Sidebar filters
st.sidebar.header("Filters")
all_products = sorted(master["product_name"].unique())
selected_products = st.sidebar.multiselect("Products", all_products, default=all_products)
selected_polarities = st.sidebar.multiselect("Polarity", ["Positive", "Negative", "Neutral"],
                                              default=["Positive", "Negative", "Neutral"])
top_n = st.sidebar.slider("Top N aspects", 5, 30, 15)

filtered = master[
    master["product_name"].isin(selected_products) &
    master["polarity"].isin(selected_polarities)
]
filtered_summary = summary[summary["product_name"].isin(selected_products)]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Top Aspects", "Heatmap", "Drill-down"])

with tab1:
    st.subheader("Sentiment breakdown per product")
    counts = filtered.groupby(["product_name", "polarity"]).size().reset_index(name="count")
    fig = px.bar(counts, x="product_name", y="count", color="polarity",
                 color_discrete_map=COLORS, barmode="group")
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Top {top_n} aspects by mention count")
    top_aspects = (filtered.groupby("aspect").size()
                   .sort_values(ascending=False).head(top_n).index.tolist())
    df2 = filtered[filtered["aspect"].isin(top_aspects)]
    counts2 = df2.groupby(["aspect", "polarity"]).size().reset_index(name="count")
    fig2 = px.bar(counts2, x="count", y="aspect", color="polarity",
                  color_discrete_map=COLORS, barmode="stack", orientation="h")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Sentiment score heatmap (Positive − Negative)")
    top_aspects_h = (filtered_summary.groupby("aspect")["total"].sum()
                     .sort_values(ascending=False).head(top_n).index.tolist())
    heat_df = filtered_summary[filtered_summary["aspect"].isin(top_aspects_h)]
    pivot = heat_df.pivot_table(index="aspect", columns="product_name",
                                values="sentiment_score", fill_value=0)
    fig3 = px.imshow(pivot, color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0, aspect="auto")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Review drill-down")
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_product = st.selectbox("Product", sorted(master["product_name"].unique()))
    with col2:
        aspects_for_product = sorted(master[master["product_name"] == sel_product]["aspect"].unique())
        sel_aspect = st.selectbox("Aspect", aspects_for_product)
    with col3:
        sel_polarity = st.selectbox("Polarity", ["Positive", "Negative", "Neutral"])

    drill = master[
        (master["product_name"] == sel_product) &
        (master["aspect"] == sel_aspect) &
        (master["polarity"] == sel_polarity)
    ][["review_snippet", "opinion"]].drop_duplicates()

    st.write(f"**{len(drill)} matching triplets**")
    st.dataframe(drill.style.applymap(
        lambda _: f"background-color: {COLORS[sel_polarity]}22",
        subset=["opinion"]
    ), use_container_width=True)
