#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para crear el esquema de base de datos para el sistema de análisis financiero.
Genera todas las tablas necesarias según el diseño definido.
"""

import os
import logging
import argparse
from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/database_schema.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('database_schema')

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
DB_TYPE = os.getenv('DB_TYPE', 'sqlite')
DB_HOST = os.getenv('DB_HOST', '')
DB_PORT = os.getenv('DB_PORT', '')
DB_NAME = os.getenv('DB_NAME', 'financial_etl.db')
DB_USER = os.getenv('DB_USER', '')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Construir URL de conexión según el tipo de base de datos
if DB_TYPE.lower() == 'sqlite':
    DB_URL = f"sqlite:///{DB_NAME}"
elif DB_TYPE.lower() == 'postgresql':
    DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
elif DB_TYPE.lower() == 'mysql':
    DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    logger.error(f"Tipo de base de datos no soportado: {DB_TYPE}")
    raise ValueError(f"Tipo de base de datos no soportado: {DB_TYPE}")

# Crear base para las clases de mapeo ORM
Base = declarative_base()

# Definición de modelos
class Asset(Base):
    """
    Tabla de activos financieros (acciones, criptomonedas, índices).
    """
    __tablename__ = 'asset'
    
    asset_id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(100), nullable=False)
    asset_type = Column(String(20), nullable=False)  # 'stock', 'crypto', 'index', etc.
    exchange = Column(String(50))
    sector = Column(String(50))
    industry = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relaciones
    stock_prices = relationship("StockPrice", back_populates="asset")
    technical_indicators = relationship("TechnicalIndicator", back_populates="asset")
    financial_ratios = relationship("FinancialRatio", back_populates="asset")
    income_statements = relationship("IncomeStatement", back_populates="asset")
    balance_sheets = relationship("BalanceSheet", back_populates="asset")
    cash_flows = relationship("CashFlow", back_populates="asset")
    crypto_market_data = relationship("CryptoMarketData", back_populates="asset")
    crypto_network_data = relationship("CryptoNetworkData", back_populates="asset")
    
    def __repr__(self):
        return f"<Asset(symbol='{self.symbol}', name='{self.name}', type='{self.asset_type}')>"


class StockPrice(Base):
    """
    Tabla de precios históricos de acciones e índices.
    """
    __tablename__ = 'stock_price'
    
    price_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    trade_date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float, nullable=False)
    adjusted_close = Column(Float)
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="stock_prices")
    
    # Índice compuesto para asset_id y trade_date
    __table_args__ = (sa.UniqueConstraint('asset_id', 'trade_date', name='uix_stock_price_asset_date'),)
    
    def __repr__(self):
        return f"<StockPrice(asset_id={self.asset_id}, date='{self.trade_date}', close={self.close_price})>"


class TechnicalIndicator(Base):
    """
    Tabla de indicadores técnicos calculados.
    """
    __tablename__ = 'technical_indicator'
    
    indicator_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    calc_date = Column(DateTime, nullable=False)
    indicator_type = Column(String(50), nullable=False)  # 'SMA_20', 'RSI_14', etc.
    value = Column(Float, nullable=False)
    parameter1 = Column(Float)  # Parámetro opcional 1
    parameter2 = Column(Float)  # Parámetro opcional 2
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="technical_indicators")
    
    # Índice compuesto para asset_id, calc_date y indicator_type
    __table_args__ = (sa.UniqueConstraint('asset_id', 'calc_date', 'indicator_type', 
                                         name='uix_technical_indicator_asset_date_type'),)
    
    def __repr__(self):
        return f"<TechnicalIndicator(asset_id={self.asset_id}, type='{self.indicator_type}', value={self.value})>"


class FinancialRatio(Base):
    """
    Tabla de ratios financieros.
    """
    __tablename__ = 'financial_ratio'
    
    ratio_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    period = Column(String(10), nullable=False)  # 'annual', 'quarterly'
    report_date = Column(DateTime, nullable=False)
    ratio_type = Column(String(50), nullable=False)  # 'PE', 'ROE', etc.
    value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="financial_ratios")
    
    # Índice compuesto
    __table_args__ = (sa.UniqueConstraint('asset_id', 'report_date', 'ratio_type', 
                                         name='uix_financial_ratio_asset_date_type'),)
    
    def __repr__(self):
        return f"<FinancialRatio(asset_id={self.asset_id}, type='{self.ratio_type}', value={self.value})>"


class IncomeStatement(Base):
    """
    Tabla de estados de resultados.
    """
    __tablename__ = 'income_statement'
    
    statement_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    period = Column(String(10), nullable=False)  # 'annual', 'quarterly'
    report_date = Column(DateTime, nullable=False)
    revenue = Column(Float)
    cost_of_revenue = Column(Float)
    gross_profit = Column(Float)
    operating_expense = Column(Float)
    operating_income = Column(Float)
    net_income = Column(Float)
    eps = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="income_statements")
    
    # Índice compuesto
    __table_args__ = (sa.UniqueConstraint('asset_id', 'report_date', 'period', 
                                         name='uix_income_statement_asset_date_period'),)
    
    def __repr__(self):
        return f"<IncomeStatement(asset_id={self.asset_id}, date='{self.report_date}', revenue={self.revenue})>"


class BalanceSheet(Base):
    """
    Tabla de balances generales.
    """
    __tablename__ = 'balance_sheet'
    
    sheet_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    period = Column(String(10), nullable=False)  # 'annual', 'quarterly'
    report_date = Column(DateTime, nullable=False)
    total_assets = Column(Float)
    current_assets = Column(Float)
    cash_equivalents = Column(Float)
    total_liabilities = Column(Float)
    current_liabilities = Column(Float)
    total_debt = Column(Float)
    total_equity = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="balance_sheets")
    
    # Índice compuesto
    __table_args__ = (sa.UniqueConstraint('asset_id', 'report_date', 'period', 
                                         name='uix_balance_sheet_asset_date_period'),)
    
    def __repr__(self):
        return f"<BalanceSheet(asset_id={self.asset_id}, date='{self.report_date}', assets={self.total_assets})>"


class CashFlow(Base):
    """
    Tabla de estados de flujo de efectivo.
    """
    __tablename__ = 'cash_flow'
    
    flow_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    period = Column(String(10), nullable=False)  # 'annual', 'quarterly'
    report_date = Column(DateTime, nullable=False)
    operating_cash_flow = Column(Float)
    capital_expenditure = Column(Float)
    free_cash_flow = Column(Float)
    dividend_paid = Column(Float)
    net_borrowings = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="cash_flows")
    
    # Índice compuesto
    __table_args__ = (sa.UniqueConstraint('asset_id', 'report_date', 'period', 
                                         name='uix_cash_flow_asset_date_period'),)
    
    def __repr__(self):
        return f"<CashFlow(asset_id={self.asset_id}, date='{self.report_date}', ocf={self.operating_cash_flow})>"


class CryptoMarketData(Base):
    """
    Tabla de datos de mercado para criptomonedas.
    """
    __tablename__ = 'crypto_market_data'
    
    crypto_data_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    trade_date = Column(DateTime, nullable=False)
    price_usd = Column(Float, nullable=False)
    market_cap = Column(Float)
    volume_24h = Column(Float)
    circulating_supply = Column(Float)
    total_supply = Column(Float)
    price_change_24h = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="crypto_market_data")
    
    # Índice compuesto
    __table_args__ = (sa.UniqueConstraint('asset_id', 'trade_date', 
                                         name='uix_crypto_market_data_asset_date'),)
    
    def __repr__(self):
        return f"<CryptoMarketData(asset_id={self.asset_id}, date='{self.trade_date}', price={self.price_usd})>"


class CryptoNetworkData(Base):
    """
    Tabla de datos de red para criptomonedas.
    """
    __tablename__ = 'crypto_network_data'
    
    network_data_id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    trade_date = Column(DateTime, nullable=False)
    hashrate = Column(Float)
    active_addresses = Column(Integer)
    avg_transaction_fee = Column(Float)
    avg_transaction_value = Column(Float)
    transaction_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relación con Asset
    asset = relationship("Asset", back_populates="crypto_network_data")
    
    # Índice compuesto
    __table_args__ = (sa.UniqueConstraint('asset_id', 'trade_date', 
                                         name='uix_crypto_network_data_asset_date'),)
    
    def __repr__(self):
        return f"<CryptoNetworkData(asset_id={self.asset_id}, date='{self.trade_date}')>"


class CorrelationData(Base):
    """
    Tabla de datos de correlación entre activos.
    """
    __tablename__ = 'correlation_data'
    
    correlation_id = Column(Integer, primary_key=True)
    asset_id_1 = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    asset_id_2 = Column(Integer, ForeignKey('asset.asset_id'), nullable=False)
    calc_date = Column(DateTime, nullable=False)
    time_window = Column(String(20), nullable=False)  # '30d', '90d', etc.
    correlation_value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Índice compuesto
    __table_args__ = (sa.UniqueConstraint('asset_id_1', 'asset_id_2', 'calc_date', 'time_window', 
                                         name='uix_correlation_data_assets_date_window'),)
    
    def __repr__(self):
        return f"<CorrelationData(assets=({self.asset_id_1},{self.asset_id_2}), value={self.correlation_value})>"


class ETLLog(Base):
    """
    Tabla de logs de procesos ETL.
    """
    __tablename__ = 'etl_log'
    
    log_id = Column(Integer, primary_key=True)
    process_name = Column(String(100), nullable=False)
    execution_time = Column(DateTime, nullable=False, default=datetime.now)
    status = Column(String(20), nullable=False)  # 'success', 'failed', etc.
    error_message = Column(String(500))
    records_processed = Column(Integer)
    execution_duration = Column(Float)  # Duración en segundos
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<ETLLog(process='{self.process_name}', status='{self.status}')>"


def create_schema(engine_url: Optional[str] = None, echo: bool = False, drop_all: bool = False):
    """
    Crea el esquema de base de datos.
    
    Args:
        engine_url: URL de conexión a la base de datos (opcional)
        echo: Si True, muestra las consultas SQL generadas
        drop_all: Si True, elimina todas las tablas existentes antes de crearlas
    """
    url = engine_url or DB_URL
    
    try:
        logger.info(f"Conectando a la base de datos: {url}")
        engine = create_engine(url, echo=echo)
        
        if drop_all:
            logger.warning("Eliminando todas las tablas existentes...")
            Base.metadata.drop_all(engine)
        
        logger.info("Creando tablas...")
        Base.metadata.create_all(engine)
        
        # Verificar que las tablas se crearon correctamente
        inspector = sa.inspect(engine)
        created_tables = inspector.get_table_names()
        expected_tables = [
            'asset', 'stock_price', 'technical_indicator', 'financial_ratio',
            'income_statement', 'balance_sheet', 'cash_flow',
            'crypto_market_data', 'crypto_network_data',
            'correlation_data', 'etl_log'
        ]
        
        missing_tables = [table for table in expected_tables if table not in created_tables]
        
        if missing_tables:
            logger.error(f"No se pudieron crear las siguientes tablas: {missing_tables}")
            return False
        else:
            logger.info(f"Esquema creado exitosamente. Tablas: {created_tables}")
            return True
    
    except Exception as e:
        logger.error(f"Error creando el esquema: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crear esquema de base de datos para sistema ETL financiero')
    parser.add_argument('--echo', action='store_true', help='Mostrar consultas SQL generadas')
    parser.add_argument('--drop', action='store_true', help='Eliminar tablas existentes antes de crearlas')
    parser.add_argument('--url', type=str, help='URL de conexión a la base de datos (opcional)')
    
    args = parser.parse_args()
    
    success = create_schema(
        engine_url=args.url,
        echo=args.echo,
        drop_all=args.drop
    )
    
    if success:
        print("Esquema de base de datos creado exitosamente.")
    else:
        print("Error creando el esquema de base de datos. Ver logs para más detalles.")