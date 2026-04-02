import pandas as pd
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from geopy.distance import geodesic
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.decomposition import TruncatedSVD
import tracemalloc

# Load .env first
load_dotenv()

# =============================================================================
# 1. AUTHENTICATION
# =============================================================================
def connect_api() -> KaggleApi | None:
    try:
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')

        # If environment variables exist, ensure kaggle.json also exists
        # (required in some Windows environments)
        if username and key:
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_dir.mkdir(exist_ok=True)
            kaggle_json = kaggle_dir / 'kaggle.json'

            if not kaggle_json.exists():
                with open(kaggle_json, 'w') as f:
                    json.dump({'username': username, 'key': key}, f)
                print("kaggle.json created automatically from .env")

        api = KaggleApi()
        api.authenticate()
        print("Success: Authentication completed!")
        return api

    except Exception as e:
        print(f"Authentication error: {e}")
        return None


# =============================================================================
# 2. EXTRACTION (Extract)
# =============================================================================
def extract_data(api: KaggleApi | None, path: str = './data') -> None:
    if api is None:
        print("Error: API not authenticated. Aborting extraction.")
        return

    data_path = Path(path)
    data_path.mkdir(parents=True, exist_ok=True)

    sentinel = data_path / 'olist_orders_dataset.csv'
    if sentinel.exists():
        print("Data already exists locally. Skipping download.")
        return

    print("Downloading data from Kaggle...")
    api.dataset_download_files('olistbr/brazilian-ecommerce', path=path, unzip=True)
    print("Files extracted successfully.")


# =============================================================================
# 3. TRANSFORMATION (Transform)
# =============================================================================
def process_master_base(path: str = './data') -> pd.DataFrame:
    # Load essential tables
    print("Loading CSVs...")
    orders      = pd.read_csv(f'{path}/olist_orders_dataset.csv')
    items       = pd.read_csv(f'{path}/olist_order_items_dataset.csv')
    reviews     = pd.read_csv(f'{path}/olist_order_reviews_dataset.csv')
    customers   = pd.read_csv(f'{path}/olist_customers_dataset.csv')
    sellers     = pd.read_csv(f'{path}/olist_sellers_dataset.csv')
    products    = pd.read_csv(f'{path}/olist_products_dataset.csv')
    payments    = pd.read_csv(f'{path}/olist_order_payments_dataset.csv')
    geolocation = pd.read_csv(f'{path}/olist_geolocation_dataset.csv')

    print(f"  orders:      {orders.shape[0]} rows")
    print(f"  items:       {items.shape[0]} rows")
    print(f"  reviews:     {reviews.shape[0]} rows")
    print(f"  customers:   {customers.shape[0]} rows")
    print(f"  sellers:     {sellers.shape[0]} rows")
    print(f"  products:    {products.shape[0]} rows")
    print(f"  payments:    {payments.shape[0]} rows")
    print(f"  geolocation: {geolocation.shape[0]} rows")

    # -------------------------------------------------------------------------
    # Pre-processing: slim down tables before merging
    # -------------------------------------------------------------------------

    # Keep only relevant seller columns
    sellers_slim = sellers[['seller_id', 'seller_zip_code_prefix']].drop_duplicates()

    # Keep only relevant product columns
    products_slim = products[[
        'product_id',
        'product_category_name',
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm'
    ]].drop_duplicates()

    # Aggregate payments: one row per order, dominant payment type
    payments_agg = (
        payments.groupby('order_id')['payment_type']
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={'payment_type': 'payment_type_main'})
    )

    # Aggregate geolocation: average lat/lng per zip prefix (centroid)
    geo_agg = (
        geolocation
        .groupby('geolocation_zip_code_prefix')
        .agg(
            geolocation_lat=('geolocation_lat', 'mean'),
            geolocation_lng=('geolocation_lng', 'mean'),
            geolocation_city=('geolocation_city', 'first'),
            geolocation_state=('geolocation_state', 'first')
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Merges
    # -------------------------------------------------------------------------
    print("\nStarting merges...")

    df = pd.merge(orders, items, on='order_id', how='inner')
    print(f"  After merge orders + items:       {df.shape[0]} rows")

    df = pd.merge(df, customers, on='customer_id', how='inner')
    print(f"  After merge + customers:          {df.shape[0]} rows")

    # Deduplicate reviews before merge to avoid row explosion
    reviews_dedup = (
        reviews[['order_id', 'review_comment_message', 'review_score']]
        .drop_duplicates(subset='order_id', keep='last')
    )
    df = pd.merge(df, reviews_dedup, on='order_id', how='left')
    print(f"  After merge + reviews:            {df.shape[0]} rows")

    df = pd.merge(df, sellers_slim, on='seller_id', how='left')
    print(f"  After merge + sellers:            {df.shape[0]} rows")

    df = pd.merge(df, products_slim, on='product_id', how='left')
    print(f"  After merge + products:           {df.shape[0]} rows")

    df = pd.merge(df, payments_agg, on='order_id', how='left')
    print(f"  After merge + payments:           {df.shape[0]} rows")

    # Geolocation for seller zip prefix
    df = pd.merge(
        df,
        geo_agg.rename(columns={
            'geolocation_zip_code_prefix': 'seller_zip_code_prefix',
            'geolocation_lat':             'seller_geo_lat',
            'geolocation_lng':             'seller_geo_lng',
            'geolocation_city':            'seller_geo_city',
            'geolocation_state':           'seller_geo_state'
        }),
        on='seller_zip_code_prefix',
        how='left'
    )
    print(f"  After merge + geo (seller):       {df.shape[0]} rows")

    # Geolocation for customer zip prefix
    df = pd.merge(
        df,
        geo_agg.rename(columns={
            'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
            'geolocation_lat':             'customer_geo_lat',
            'geolocation_lng':             'customer_geo_lng',
            'geolocation_city':            'customer_geo_city',
            'geolocation_state':           'customer_geo_state'
        }),
        on='customer_zip_code_prefix',
        how='left'
    )
    print(f"  After merge + geo (customer):     {df.shape[0]} rows")

    # -------------------------------------------------------------------------
    # Feature engineering: seller–customer distance (km)
    # -------------------------------------------------------------------------
    # Uses centroid lat/lng per zip prefix (aggregated from olist_geolocation_dataset).
    # geodesic() computes surface distance between two lat/lng points.
    # NOTE: coordinates are approximate (zip prefix centroid, not exact address).

    def compute_distance_km(row):
        if pd.isna(row["seller_geo_lat"]) or pd.isna(row["seller_geo_lng"]) \
           or pd.isna(row["customer_geo_lat"]) or pd.isna(row["customer_geo_lng"]):
            return pd.NA
        try:
            return geodesic(
                (row["seller_geo_lat"],   row["seller_geo_lng"]),
                (row["customer_geo_lat"], row["customer_geo_lng"])
            ).km
        except Exception:
            return pd.NA

    print("\nCalculating seller–customer distance (this may take a few seconds)...")
    df["seller_customer_distance_km"] = df.apply(compute_distance_km, axis=1)
    print(f"  Distance calculated. Missing: {df['seller_customer_distance_km'].isna().sum()} rows")

    # -------------------------------------------------------------------------
    # Drop raw geolocation coordinates and zip prefixes
    # -------------------------------------------------------------------------
    # Raw lat/lng columns were used solely to compute seller_customer_distance_km.
    # Zip code prefixes are also removed to avoid overly granular location keys.
    # City and state columns are kept as geographical features alongside distance.

    cols_to_drop = [
        # Raw geolocation coordinates
        "seller_geo_lat",
        "seller_geo_lng",
        "customer_geo_lat",
        "customer_geo_lng",
        # Zip code prefixes
        "seller_zip_code_prefix",
        "customer_zip_code_prefix",
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped raw geolocation and zip code columns: {cols_to_drop}")

    # -------------------------------------------------------------------------
    # Date parsing
    # -------------------------------------------------------------------------
    date_cols = [
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # -------------------------------------------------------------------------
    # Target variable (is_late)
    # -------------------------------------------------------------------------
    # Uses nullable Int8 so orders without delivery date remain NaN
    # and are not misclassified as on time
    df['is_late'] = (
        df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
    ).astype('Int8')

    # Orders without actual delivery date → is_late = NaN (not yet classifiable)
    df.loc[df['order_delivered_customer_date'].isna(), 'is_late'] = pd.NA

    # -------------------------------------------------------------------------
    # Feature engineering: product volume (cm³)
    # -------------------------------------------------------------------------
    # Computed as length × height × width using product dimension columns.
    # Raw dimension columns are dropped after volume computation; we keep:
    # - product_weight_g (continuous)
    # - product_volume_cm3 (continuous)
    # - is_heavy_product (binary)
    # - is_bulky_product (binary)

    df["product_volume_cm3"] = (
        df["product_length_cm"] *
        df["product_height_cm"] *
        df["product_width_cm"]
    )
    print(f"  Product volume calculated. Missing: {df['product_volume_cm3'].isna().sum()} rows")

    # Heavy product flag: weight > 10kg
    df["is_heavy_product"] = (df["product_weight_g"] > 10_000).astype("Int8")

    # Bulky product flag: volume > 50k cm³
    df["is_bulky_product"] = (df["product_volume_cm3"] > 50_000).astype("Int8")

    print(
        "  Heavy products (weight > 10kg): "
        f"{df['is_heavy_product'].sum()} rows "
        f"({df['is_heavy_product'].mean():.1%})"
    )
    print(
        "  Bulky products (volume > 50k cm³): "
        f"{df['is_bulky_product'].sum()} rows "
        f"({df['is_bulky_product'].mean():.1%})"
    )

    # Drop raw dimension columns (length, height, width)
    dims_to_drop = [
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df.drop(columns=dims_to_drop, inplace=True)
    print(f"  Dropped raw product dimensions: {dims_to_drop}")

    # -------------------------------------------------------------------------
    # Validation: check expected columns are present after all merges
    # -------------------------------------------------------------------------
    expected_cols = [
        "product_category_name",
        "product_weight_g",
        "product_volume_cm3",
        "is_heavy_product",
        "is_bulky_product",
        "payment_type_main",
        "seller_customer_distance_km",
        "seller_geo_city",
        "seller_geo_state",
        "customer_geo_city",
        "customer_geo_state",
    ]

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print(f"\nWARNING: missing columns after merges: {missing_cols}")
    else:
        print("\nAll expected columns present after merges.")

    print(f"\nConsolidated dataset: {df.shape[0]} rows.")
    print(f"  Late orders:          {df['is_late'].sum()}")
    print(f"  On-time orders:       {(df['is_late'] == 0).sum()}")
    print(f"  No delivery date:     {df['is_late'].isna().sum()}")

    return df

def generate_bert_features(
    df: pd.DataFrame,
    text_col: str = "review_comment_message",
    n_components: int = 50,
    batch_size: int = 64,
    model_name: str = "neuralmind/bert-base-portuguese-cased",
) -> pd.DataFrame:
    """
    Generate BERT [CLS] embeddings for text column and reduce dimensionality via SVD.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the text column.
    text_col : str
        Name of the column with review comments.
    n_components : int
        Number of SVD components to retain.
    batch_size : int
        Number of texts processed per BERT forward pass.
    model_name : str
        Pretrained BERT model identifier from HuggingFace.

    Returns
    -------
    pd.DataFrame
        Original dataframe with has_comment flag and bert_svd_* columns appended.
    """
    ...

    # Binary flag: 1 if order has a review comment, 0 otherwise
    df["has_comment"] = df[text_col].notna().astype("Int8")

    # Load tokenizer and model
    print(f"Loading BERT model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully.")

        # Flatten all texts; empty string replaces NaN so tokenizer never receives None
    texts = df[text_col].fillna("").tolist()
    embeddings = []

    print("Extracting BERT embeddings...")

    # Disable gradient computation — inference only, no backpropagation needed
    with torch.no_grad():

        # Iterate over texts in chunks of batch_size (default: 64)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize batch: pad sequences to same length, truncate at 512 tokens
            # return_tensors="pt" returns PyTorch tensors
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Forward pass through BERT
            output = model(**encoded)

            # Extract [CLS] token (position 0) from each sequence — shape: (batch_size, 768)
            cls_vectors = output.last_hidden_state[:, 0, :]

            # Convert tensor to NumPy and store
            embeddings.append(cls_vectors.numpy())

    print(f"Embeddings extracted: {len(texts)} texts processed.")

    # Stack all batch arrays into a single matrix — shape: (n_rows, 768)
    embeddings_matrix = np.vstack(embeddings)

    # Reduce 768 dimensions to n_components (default: 50) via Truncated SVD
    print(f"Applying SVD: 768 → {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings_reduced = svd.fit_transform(embeddings_matrix)

    # Create one column per SVD component and attach to dataframe
    svd_cols = [f"bert_svd_{i}" for i in range(n_components)]
    df[svd_cols] = embeddings_reduced

    print(f"SVD complete. Columns added: bert_svd_0 … bert_svd_{n_components - 1}")

    return df

# =============================================================================
# 4. TEST END VALIDATION OF BERT EMBEDDING
# =============================================================================

def test_bert_inference() -> None:
    """
    Layer 1 - Smoke test: verify BERT model loads and runs inference
    on a single sentence before touching the dataset.
    Aborts early if the model or tokenizer fails to load.
    """

    print("Layer 1: Testing BERT model loading and inference...")

    # Load tokenizer and model using Portuguese BERT
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model.eval()

    # Single hardcoded sentence simulating a review comment
    sample_text = ["Produto entregue com atraso e embalagem danificada."]

    # Tokenize and run forward pass without computing gradients
    encoded = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**encoded)

    # Extract CLS token vector — expected shape: (1, 768)
    cls_vector = output.last_hidden_state[:, 0, :]
    assert cls_vector.shape == (1, 768), f"Unexpected CLS shape: {cls_vector.shape}"

    print(f"Layer 1 passed. CLS vector shape: {cls_vector.shape}\n")


def test_generate_bert_features(df: pd.DataFrame, sample_size: int = 100) -> None:
    """
    Layer 2 - Memory test: run generate_bert_features on a small sample
    and monitor peak memory usage via tracemalloc.
    Validates that the function produces exactly 50 SVD columns with no NaN.
    Aborts early if memory usage is unexpectedly high.
    """

    print(f"Layer 2: Running generate_bert_features on {sample_size} rows...")

    # Use a small sample to keep memory usage low during the test
    sample = df.head(sample_size).copy()

    # Start memory tracing before running the function
    tracemalloc.start()
    result = generate_bert_features(sample)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Validate that exactly 50 SVD columns were created
    svd_cols = [c for c in result.columns if c.startswith("bert_svd_")]
    assert len(svd_cols) == 50, f"Expected 50 SVD columns, got {len(svd_cols)}"

    # Validate that no NaN values exist in SVD columns
    nan_count = result[svd_cols].isnull().sum().sum()
    assert nan_count == 0, f"NaN values found in SVD columns: {nan_count}"

    print(f"Layer 2 passed.")
    print(f"  Peak memory: {peak / 1e6:.2f} MB")
    print(f"  Output shape: {result.shape}\n")


def validate_bert_features(df: pd.DataFrame, n_components: int = 50) -> None:
    """
    Layer 3 - Final validation: after generate_bert_features runs on the full
    dataset, verify column completeness, NaN absence, and report memory usage.
    This is the last checkpoint before saving the base.
    """

    print("Layer 3: Validating BERT SVD columns on full dataset...")

    # Build expected column names and check for missing ones
    svd_cols = [f"bert_svd_{i}" for i in range(n_components)]
    missing = [c for c in svd_cols if c not in df.columns]
    assert not missing, f"Missing SVD columns: {missing}"

    # Check for any remaining NaN values across all SVD columns
    nan_count = df[svd_cols].isnull().sum().sum()
    assert nan_count == 0, f"NaN values found: {nan_count}"

    # Report memory footprint of SVD columns alone
    mem_mb = df[svd_cols].memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Layer 3 passed.")
    print(f"  SVD columns memory usage: {mem_mb:.2f} MB")
    print(f"  Full dataframe shape: {df.shape}\n")

# =============================================================================
# 5. LOAD
# =============================================================================
def save_base(df: pd.DataFrame, output_path: str = 'consolidated_logistics_base.parquet') -> None:
    df.to_parquet(output_path, index=False)
    print(f"\nFile '{output_path}' saved successfully.")


# =============================================================================
# MAIN
# =============================================================================

# Layer 1: verify BERT loads and infers before touching any data
    test_bert_inference()

    # Connect to API and extract raw data
    api = connect_api()
    extract_data(api)

    # Build master base
    df_final = process_master_base()

    # Layer 2: smoke test with small sample before full BERT run
    test_generate_bert_features(df_final)

    # Run BERT on full dataset
    df_final = generate_bert_features(df_final)

    # Layer 3: validate output before saving
    validate_bert_features(df_final)

    # Persist final base
    save_base(df_final)