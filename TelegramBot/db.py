import pandas as pd
import re
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Загрузка переменных окружения из файла .env
load_dotenv()

# Параметры подключения из переменных окружения
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
server = os.getenv('DB_SERVER')
port = os.getenv('DB_PORT')
database = os.getenv('DB_DATABASE')
driver = os.getenv('DB_DRIVER')

# Создание движка SQLAlchemy
engine = create_engine(
    f"mssql+pyodbc://{username}:{password}@{server}:{port}/{database}?driver={driver.replace(' ', '+')}"
)

# Определение полного имени таблицы
table_schema = 'dbo'
table_name = '_see'
full_table_name = f"{table_schema}.{table_name}"

# Список интересующих нас столбцов
columns_select = ', '.join([
    'TypeName', 'Type2', 'Type', 'FhaseCount',
    'PrimaryAmperage', 'NomonalLoad', 'ManufacturerYear',
    'GosNumber', 'SearchKey'
])

def clean_input(input_string):
    """Очистка входной строки от пробелов и специальных символов."""
    return re.sub(r'\W+', '', input_string)

def search_by_type(search_type):
    """Выполняет первый запрос к базе данных."""
    query = f"""
    SELECT {columns_select}
    FROM {full_table_name}
    WHERE SearchKey LIKE '%{search_type}%';
    """
    df = pd.read_sql_query(query, engine)

    if not df.empty:
        total_count = len(df)
        unique_gos_numbers = ', '.join(df['GosNumber'].dropna().unique())
        return total_count, unique_gos_numbers, df
    else:
        return 0, None, pd.DataFrame()

def search_by_mode(search_mode):
    """Выполняет второй запрос к базе данных."""
    query = f"""
    SELECT {columns_select}
    FROM {full_table_name}
    WHERE SearchKey LIKE '%{search_mode}%';
    """
    df = pd.read_sql_query(query, engine)

    if not df.empty:
        total_count = len(df)
        unique_gos_numbers = ', '.join(df['GosNumber'].dropna().unique())
        return total_count, unique_gos_numbers, df
    else:
        return 0, None, pd.DataFrame()

def search_by_year(search_year):
    """Выполняет запрос к базе данных по году."""
    query = f"""
    SELECT {columns_select}
    FROM {full_table_name}
    WHERE ManufacturerYear = {search_year};
    """
    df = pd.read_sql_query(query, engine)

    if not df.empty:
        total_count = len(df)
        unique_gos_numbers = ', '.join(df['GosNumber'].dropna().unique())
        return total_count, unique_gos_numbers, df
    else:
        return 0, None, pd.DataFrame()