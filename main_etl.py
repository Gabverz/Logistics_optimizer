import pandas as pd
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

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
    # Carregando as tabelas essenciais
    print("Carregando CSVs...")
    orders    = pd.read_csv(f'{path}/olist_orders_dataset.csv')
    items     = pd.read_csv(f'{path}/olist_order_items_dataset.csv')
    reviews   = pd.read_csv(f'{path}/olist_order_reviews_dataset.csv')
    customers = pd.read_csv(f'{path}/olist_customers_dataset.csv')

    print(f"  orders:    {orders.shape[0]} linhas")
    print(f"  items:     {items.shape[0]} linhas")
    print(f"  reviews:   {reviews.shape[0]} linhas")
    print(f"  customers: {customers.shape[0]} linhas")

    # -------------------------------------------------------------------------
    # Merges
    # -------------------------------------------------------------------------
    print("\nIniciando Merges...")

    df = pd.merge(orders, items, on='order_id', how='inner')
    print(f"  Após merge orders + items:     {df.shape[0]} linhas")

    df = pd.merge(df, customers, on='customer_id', how='inner')
    print(f"  Após merge + customers:        {df.shape[0]} linhas")

    # Deduplica reviews antes do merge para evitar multiplicação de linhas
    reviews_dedup = (
        reviews[['order_id', 'review_comment_message', 'review_score']]
        .drop_duplicates(subset='order_id', keep='last')
    )
    df = pd.merge(df, reviews_dedup, on='order_id', how='left')
    print(f"  Após merge + reviews:          {df.shape[0]} linhas")

    # -------------------------------------------------------------------------
    # Tratamento de datas
    # -------------------------------------------------------------------------
    cols_data = [
        'order_purchase_timestamp',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in cols_data:
        df[col] = pd.to_datetime(df[col])

    # -------------------------------------------------------------------------
    # Criação da variável-alvo (is_late)
    # -------------------------------------------------------------------------
    # Usa Int8 nullable para que pedidos sem data de entrega fiquem como NaN
    # e não sejam classificados erroneamente como "no prazo"
    df['is_late'] = (
        df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
    ).astype('Int8')

    # Pedidos sem data de entrega real → is_late = NaN (não classificável ainda)
    df.loc[df['order_delivered_customer_date'].isna(), 'is_late'] = pd.NA

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