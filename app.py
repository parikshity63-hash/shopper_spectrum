"""
Shopper Spectrum — Customer Segmentation & Product Recommendation App
Run with: streamlit run app.py

Expected files in the same folder (adjust paths in CONFIG below if different):
    models/kmeans_model.pkl
    models/scaler.pkl
    models/similarity_df.pkl
    data/customer_segments.csv
    data/product_reference.csv   (optional — generate with generate_product_reference.py)
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------------
# CONFIG — adjust these paths to match where you keep your files
# ------------------------------------------------------------------
MODEL_DIR = Path("models")
DATA_DIR = Path("data")

KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
SIMILARITY_PATH = MODEL_DIR / "similarity_df.pkl"
SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"
PRODUCT_REF_PATH = DATA_DIR / "product_reference.csv"  # optional

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
)

# ------------------------------------------------------------------
# LOADERS (cached so files are only read/loaded once)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    similarity_df = joblib.load(SIMILARITY_PATH)
    return kmeans, scaler, similarity_df


@st.cache_data
def load_segments():
    rfm = pd.read_csv(SEGMENTS_PATH)
    rfm["CustomerID"] = rfm["CustomerID"].astype(float).astype(int)
    return rfm


@st.cache_data
def load_product_reference():
    if PRODUCT_REF_PATH.exists():
        ref = pd.read_csv(PRODUCT_REF_PATH)
        return ref
    return None


@st.cache_data
def compute_segment_names(rfm: pd.DataFrame) -> dict:
    """
    Dynamically maps each Cluster ID to a human-readable segment name
    based on its average Recency / Frequency / Monetary behavior,
    rather than hardcoding cluster numbers (cluster IDs can shift
    if the model is retrained).
    """
    summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Lower Recency = better. Higher Frequency/Monetary = better.
    summary["RecencyRank"] = summary["Recency"].rank(ascending=True)
    summary["FrequencyRank"] = summary["Frequency"].rank(ascending=False)
    summary["MonetaryRank"] = summary["Monetary"].rank(ascending=False)
    summary["Score"] = (
        summary["RecencyRank"] + summary["FrequencyRank"] + summary["MonetaryRank"]
    )
    summary = summary.sort_values("Score")

    label_pool = ["Champions", "Loyal Customers", "Occasional Shoppers", "At Risk"]
    n_clusters = len(summary)
    if n_clusters <= len(label_pool):
        labels = label_pool[:n_clusters]
    else:
        labels = label_pool + [f"Segment {i}" for i in range(len(label_pool), n_clusters)]

    return dict(zip(summary.index, labels))


kmeans, scaler, similarity_df = load_models()
rfm_df = load_segments()
product_ref = load_product_reference()
segment_names = compute_segment_names(rfm_df)

SEGMENT_COLORS = {
    "Champions": "#2ECC71",
    "Loyal Customers": "#3498DB",
    "Occasional Shoppers": "#F1C40F",
    "At Risk": "#E74C3C",
}

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("🛍️ Shopper Spectrum")
page = st.sidebar.radio(
    "Navigate",
    ["Customer Segmentation", "Product Recommendations"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Customer segmentation via RFM + KMeans clustering, "
    "and item-based product recommendations via cosine similarity."
)

# ==================================================================
# PAGE 1 — CUSTOMER SEGMENTATION DASHBOARD
# ==================================================================
if page == "Customer Segmentation":
    st.title("Customer Segmentation Dashboard")
    st.write("Enter a Customer ID to view their RFM profile and segment.")

    col_input, _ = st.columns([1, 2])
    with col_input:
        customer_id_input = st.text_input("Customer ID", placeholder="e.g. 12347")
        search = st.button("Search", type="primary")

    if search and customer_id_input.strip():
        try:
            customer_id = int(float(customer_id_input.strip()))
        except ValueError:
            st.error("Please enter a valid numeric Customer ID.")
            customer_id = None

        if customer_id is not None:
            record = rfm_df[rfm_df["CustomerID"] == customer_id]

            if record.empty:
                st.warning(f"No customer found with ID **{customer_id}**.")
            else:
                row = record.iloc[0]
                cluster = int(row["Cluster"])
                segment = segment_names.get(cluster, f"Segment {cluster}")
                color = SEGMENT_COLORS.get(segment, "#95A5A6")

                st.markdown(
                    f"""
                    <div style="padding:14px 18px;border-radius:10px;
                    background-color:{color}22;border:1px solid {color};margin-bottom:16px;">
                        <span style="font-size:1.1rem;">Segment: </span>
                        <span style="font-size:1.3rem;font-weight:700;color:{color};">
                        {segment}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Recency (days)", f"{row['Recency']:.0f}")
                m2.metric("Frequency (orders)", f"{row['Frequency']:.0f}")
                m3.metric("Monetary (spend)", f"£{row['Monetary']:.2f}")
                m4.metric("Cluster ID", cluster)

    st.markdown("---")
    st.subheader("Customer Distribution Across Clusters")

    dist = rfm_df["Cluster"].value_counts().sort_index().reset_index()
    dist.columns = ["Cluster", "Count"]
    dist["Segment"] = dist["Cluster"].map(segment_names)

    fig = px.bar(
        dist,
        x="Segment",
        y="Count",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        text="Count",
        title="Number of Customers per Segment",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of Customers")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# PAGE 2 — PRODUCT RECOMMENDATION SYSTEM
# ==================================================================
else:
    st.title("Product Recommendation System")
    st.write("Search for a product by name or StockCode to see similar products.")

    search_mode = st.radio("Search by", ["Product Name", "StockCode"], horizontal=True)
    query = st.text_input("Type to search", placeholder="Start typing...")

    matched_description = None

    if query.strip():
        if search_mode == "Product Name":
            candidates = [d for d in similarity_df.index if query.lower() in str(d).lower()]
        else:
            if product_ref is None:
                st.error(
                    "StockCode search requires `product_reference.csv`, which wasn't found. "
                    "Run `generate_product_reference.py` in your notebook to create it, "
                    "or search by Product Name instead."
                )
                candidates = []
            else:
                sub = product_ref[
                    product_ref["StockCode"].astype(str).str.lower().str.contains(query.lower())
                ]
                candidates = sub["Description"].tolist()

        # keep only candidates that actually exist in the similarity matrix
        candidates = [c for c in candidates if c in similarity_df.index]

        if candidates:
            matched_description = st.selectbox(
                f"Matching products ({len(candidates)} found)", sorted(set(candidates))
            )
        else:
            st.info("No matching products found in the similarity matrix.")

    if matched_description:
        st.markdown("---")
        st.subheader(f"Top 5 products similar to: *{matched_description}*")

        sims = similarity_df[matched_description].sort_values(ascending=False)
        sims = sims.drop(labels=[matched_description], errors="ignore")
        top5 = sims.head(5)

        results = pd.DataFrame({
            "Product Name": top5.index,
            "Similarity Score": top5.values.round(3),
        })

        if product_ref is not None:
            results = results.merge(
                product_ref, left_on="Product Name", right_on="Description", how="left"
            ).drop(columns=["Description"])
            results = results[["Product Name", "StockCode", "UnitPrice", "Similarity Score"]]
        else:
            st.caption(
                "StockCode / UnitPrice unavailable — add `product_reference.csv` to show them."
            )

        st.dataframe(results, use_container_width=True, hide_index=True)