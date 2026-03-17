import pandas as pd
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from geopy.distance import geodesic

# Carrega o .env primeiro
load_dotenv()

# =============================================================================
# 1. AUTENTICAÇÃO
# =============================================================================
def conectar_api():
    try:
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')

        # Se as variáveis de ambiente existirem, garante que o kaggle.json
        # também existe (necessário em alguns ambientes Windows)
        if username and key:
            kaggle_dir = Path.home() / '.kaggle'
            kaggle_dir.mkdir(exist_ok=True)
            kaggle_json = kaggle_dir / 'kaggle.json'

            if not kaggle_json.exists():
                with open(kaggle_json, 'w') as f:
                    json.dump({'username': username, 'key': key}, f)
                print("kaggle.json criado automaticamente a partir do .env")

        api = KaggleApi()
        api.authenticate()
        print("Sucesso: Autenticação concluída!")
        return api

    except Exception as e:
        print(f"Erro ao autenticar: {e}")
        return None


# =============================================================================
# 2. EXTRAÇÃO (Extract)
# =============================================================================
def extrair_dados(api, path='./data'):
    # Recebe api como parâmetro (sem depender de variável global)
    if api is None:
        print("Erro: API não autenticada. Abortando extração.")
        return

    # Cria o diretório se não existir
    if not os.path.exists(path):
        os.makedirs(path)

    # Evita re-download se os dados já existem
    csv_exemplo = f'{path}/olist_orders_dataset.csv'
    if os.path.exists(csv_exemplo):
        print("Dados já existem localmente. Pulando download.")
        return

    print("Baixando dados do Kaggle...")
    api.dataset_download_files('olistbr/brazilian-ecommerce', path=path, unzip=True)
    print("Arquivos extraídos com sucesso.")


# =============================================================================
# 3. TRANSFORMAÇÃO (Transform)
# =============================================================================

def processar_base_mestre(path='./data'):
    # Load essential tables
    print("Carregando CSVs...")
    orders      = pd.read_csv(f'{path}/olist_orders_dataset.csv')
    items       = pd.read_csv(f'{path}/olist_order_items_dataset.csv')
    reviews     = pd.read_csv(f'{path}/olist_order_reviews_dataset.csv')
    customers   = pd.read_csv(f'{path}/olist_customers_dataset.csv')
    sellers     = pd.read_csv(f'{path}/olist_sellers_dataset.csv')
    products    = pd.read_csv(f'{path}/olist_products_dataset.csv')
    payments    = pd.read_csv(f'{path}/olist_order_payments_dataset.csv')
    geolocation = pd.read_csv(f'{path}/olist_geolocation_dataset.csv')

    print(f"  orders:      {orders.shape[0]} linhas")
    print(f"  items:       {items.shape[0]} linhas")
    print(f"  reviews:     {reviews.shape[0]} linhas")
    print(f"  customers:   {customers.shape[0]} linhas")
    print(f"  sellers:     {sellers.shape[0]} linhas")
    print(f"  products:    {products.shape[0]} linhas")
    print(f"  payments:    {payments.shape[0]} linhas")
    print(f"  geolocation: {geolocation.shape[0]} linhas")

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
    print("\nIniciando Merges...")

    df = pd.merge(orders, items, on='order_id', how='inner')
    print(f"  Após merge orders + items:        {df.shape[0]} linhas")

    df = pd.merge(df, customers, on='customer_id', how='inner')
    print(f"  Após merge + customers:           {df.shape[0]} linhas")

    # Deduplicate reviews before merge to avoid row explosion
    reviews_dedup = (
        reviews[['order_id', 'review_comment_message', 'review_score']]
        .drop_duplicates(subset='order_id', keep='last')
    )
    df = pd.merge(df, reviews_dedup, on='order_id', how='left')
    print(f"  Após merge + reviews:             {df.shape[0]} linhas")

    df = pd.merge(df, sellers_slim, on='seller_id', how='left')
    print(f"  Após merge + sellers:             {df.shape[0]} linhas")

    df = pd.merge(df, products_slim, on='product_id', how='left')
    print(f"  Após merge + products:            {df.shape[0]} linhas")

    df = pd.merge(df, payments_agg, on='order_id', how='left')
    print(f"  Após merge + payments:            {df.shape[0]} linhas")

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
    print(f"  Após merge + geo (seller):        {df.shape[0]} linhas")

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
    print(f"  Após merge + geo (customer):      {df.shape[0]} linhas")

    # -------------------------------------------------------------------------
    # Defining seller–customer distance (km)
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

    df["seller_customer_distance_km"] = df.apply(compute_distance_km, axis=1)
    print(f"  Distance calculated. Missing: {df['seller_customer_distance_km'].isna().sum()} rows")

    # -------------------------------------------------------------------------
    # Drop geolocation columns
    # -------------------------------------------------------------------------
    # Geolocation lat/lng columns (seller and customer) were used solely to
    # compute seller_customer_distance_km above. They are dropped here to keep
    # the consolidated base lean and avoid leakage of raw coordinates into the
    # model. Geography is represented by: customer_state, seller_state (already
    # present) and seller_customer_distance_km (engineered above).
    # Zip code prefix columns are also dropped as they are granular proxies of
    # location already captured by state and distance features.

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
    print(f"  Dropped geolocation and zip code columns: {cols_to_drop}")


    # -------------------------------------------------------------------------
    # Date parsing
    # -------------------------------------------------------------------------
    cols_data = [
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in cols_data:
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
    # Raw dimension columns are kept as individual features alongside volume.

    df["product_volume_cm3"] = (
        df["product_length_cm"] *
        df["product_height_cm"] *
        df["product_width_cm"]
    )
    print(f"  Product volume calculated. Missing: {df['product_volume_cm3'].isna().sum()} rows")

    # -------------------------------------------------------------------------
    # Validation: check expected columns are present after all merges
    # -------------------------------------------------------------------------
    expected_cols = [
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_volume_cm3",
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

    print(f"\nDataset consolidado com {df.shape[0]} linhas.")
    print(f"  Pedidos atrasados:     {df['is_late'].sum()}")
    print(f"  Pedidos no prazo:      {(df['is_late'] == 0).sum()}")
    print(f"  Sem data de entrega:   {df['is_late'].isna().sum()}")

    return df


# =============================================================================
# 4. CARGA (Load)
# =============================================================================
def salvar_base(df, output_path='base_consolidada_logistica.parquet'):
    df.to_parquet(output_path, index=False)
    print(f"\nArquivo '{output_path}' gerado com sucesso.")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    api = conectar_api()
    extrair_dados(api)
    df_final = processar_base_mestre()
    salvar_base(df_final)