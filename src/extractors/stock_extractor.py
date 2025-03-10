#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo extractor de datos bursátiles.
Obtiene datos históricos de precios e indicadores para acciones e índices.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

import requests
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/stock_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stock_extractor')

# Cargar variables de entorno
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
MAX_RETRIES = 3
RETRY_DELAY = 12  # Segundos entre reintentos (respeta límites de API)


class StockDataExtractor:
    """
    Extractor de datos de acciones utilizando Alpha Vantage como fuente principal
    y Yahoo Finance como respaldo.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el extractor con clave API opcional.
        
        Args:
            api_key: Clave de API de Alpha Vantage (opcional si se define en .env)
        """
        self.alpha_api_key = api_key or ALPHA_VANTAGE_API_KEY
        
        if not self.alpha_api_key:
            logger.warning("No se proporcionó API key para Alpha Vantage. Algunas funcionalidades serán limitadas.")
        
        self.alpha_ts = TimeSeries(key=self.alpha_api_key, output_format='pandas')
        self.request_count = 0
        self.last_request_time = datetime.now() - timedelta(minutes=1)
    
    def _respect_rate_limit(self) -> None:
        """
        Asegura que se respeten los límites de tasa de Alpha Vantage (5 llamadas/min en plan gratuito).
        """
        self.request_count += 1
        now = datetime.now()
        
        if (now - self.last_request_time).seconds < 60 and self.request_count >= 5:
            sleep_time = 60 - (now - self.last_request_time).seconds
            logger.info(f"Límite de tasa alcanzado. Esperando {sleep_time} segundos...")
            time.sleep(sleep_time + 1)  # +1 para asegurar
            self.request_count = 0
            self.last_request_time = datetime.now()
        elif (now - self.last_request_time).seconds >= 60:
            self.request_count = 1
            self.last_request_time = now
    
    def get_daily_stock_data(self, symbol: str, output_size: str = 'full') -> pd.DataFrame:
        """
        Obtiene datos diarios para un símbolo de acción.
        
        Args:
            symbol: Símbolo de la acción o índice
            output_size: 'compact' (últimos 100 puntos) o 'full' (hasta 20 años de datos históricos)
            
        Returns:
            DataFrame con datos históricos diarios
        """
        logger.info(f"Obteniendo datos diarios para {symbol}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                # Intentar con Alpha Vantage primero
                if self.alpha_api_key:
                    self._respect_rate_limit()
                    data, meta_data = self.alpha_ts.get_daily(symbol=symbol, outputsize=output_size)
                    data.columns = [col.split('. ')[1] for col in data.columns]
                    data.reset_index(inplace=True)
                    data.rename(columns={'index': 'date'}, inplace=True)
                    logger.info(f"Datos obtenidos exitosamente de Alpha Vantage para {symbol}")
                    return data
                
                # Falló Alpha Vantage o no hay API key, usar Yahoo Finance
                logger.info(f"Usando respaldo Yahoo Finance para {symbol}")
                yf_ticker = yf.Ticker(symbol)
                data = yf_ticker.history(period="max" if output_size == 'full' else "3mo")
                data.reset_index(inplace=True)
                
                # Normalizar nombres de columnas para coincidir con Alpha Vantage
                data.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
                
                if 'Adj Close' in data.columns:
                    data.rename(columns={'Adj Close': 'adjusted_close'}, inplace=True)
                
                logger.info(f"Datos obtenidos exitosamente de Yahoo Finance para {symbol}")
                return data
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener datos para {symbol} después de {MAX_RETRIES} intentos.")
                    raise
        
        return pd.DataFrame()  # Nunca debería llegar aquí debido al raise anterior
    
    def get_intraday_stock_data(self, symbol: str, interval: str = '60min') -> pd.DataFrame:
        """
        Obtiene datos intradía para un símbolo de acción.
        
        Args:
            symbol: Símbolo de la acción
            interval: Intervalo de tiempo ('1min', '5min', '15min', '30min', '60min')
            
        Returns:
            DataFrame con datos intradía
        """
        if not self.alpha_api_key:
            logger.error("Se requiere API key de Alpha Vantage para datos intradía")
            raise ValueError("Alpha Vantage API key es necesaria para datos intradía")
        
        logger.info(f"Obteniendo datos intradía ({interval}) para {symbol}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit()
                data, meta_data = self.alpha_ts.get_intraday(
                    symbol=symbol,
                    interval=interval,
                    outputsize='full'
                )
                
                data.columns = [col.split('. ')[1] for col in data.columns]
                data.reset_index(inplace=True)
                data.rename(columns={'index': 'datetime'}, inplace=True)
                
                logger.info(f"Datos intradía obtenidos exitosamente para {symbol}")
                return data
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener datos intradía para {symbol} después de {MAX_RETRIES} intentos.")
                    raise
        
        return pd.DataFrame()
    
    def get_company_overview(self, symbol: str) -> Dict:
        """
        Obtiene información general de la empresa.
        
        Args:
            symbol: Símbolo de la acción
            
        Returns:
            Diccionario con información de la empresa
        """
        if not self.alpha_api_key:
            logger.error("Se requiere API key de Alpha Vantage para información de empresa")
            raise ValueError("Alpha Vantage API key es necesaria para información de empresa")
        
        logger.info(f"Obteniendo información general para {symbol}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit()
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alpha_api_key}"
                r = requests.get(url)
                data = r.json()
                
                if "Error Message" in data:
                    raise ValueError(f"Error en Alpha Vantage: {data['Error Message']}")
                
                logger.info(f"Información de empresa obtenida exitosamente para {symbol}")
                return data
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener información de empresa para {symbol} después de {MAX_RETRIES} intentos.")
                    raise
        
        return {}
    
    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """
        Obtiene el estado de resultados de la empresa.
        
        Args:
            symbol: Símbolo de la acción
            
        Returns:
            DataFrame con datos del estado de resultados
        """
        if not self.alpha_api_key:
            logger.error("Se requiere API key de Alpha Vantage para estado de resultados")
            raise ValueError("Alpha Vantage API key es necesaria para estado de resultados")
        
        logger.info(f"Obteniendo estado de resultados para {symbol}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit()
                url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={self.alpha_api_key}"
                r = requests.get(url)
                data = r.json()
                
                if "Error Message" in data:
                    raise ValueError(f"Error en Alpha Vantage: {data['Error Message']}")
                
                # Convertir a DataFrame
                if 'annualReports' in data:
                    df = pd.DataFrame(data['annualReports'])
                    df['reportType'] = 'annual'
                    
                    if 'quarterlyReports' in data:
                        df_q = pd.DataFrame(data['quarterlyReports'])
                        df_q['reportType'] = 'quarterly'
                        df = pd.concat([df, df_q])
                    
                    logger.info(f"Estado de resultados obtenido exitosamente para {symbol}")
                    return df
                else:
                    logger.warning(f"No se encontraron datos de estado de resultados para {symbol}")
                    return pd.DataFrame()
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener estado de resultados para {symbol} después de {MAX_RETRIES} intentos.")
                    raise
        
        return pd.DataFrame()
    
    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """
        Obtiene el balance general de la empresa.
        
        Args:
            symbol: Símbolo de la acción
            
        Returns:
            DataFrame con datos del balance general
        """
        if not self.alpha_api_key:
            logger.error("Se requiere API key de Alpha Vantage para balance general")
            raise ValueError("Alpha Vantage API key es necesaria para balance general")
        
        logger.info(f"Obteniendo balance general para {symbol}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit()
                url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={self.alpha_api_key}"
                r = requests.get(url)
                data = r.json()
                
                if "Error Message" in data:
                    raise ValueError(f"Error en Alpha Vantage: {data['Error Message']}")
                
                # Convertir a DataFrame
                if 'annualReports' in data:
                    df = pd.DataFrame(data['annualReports'])
                    df['reportType'] = 'annual'
                    
                    if 'quarterlyReports' in data:
                        df_q = pd.DataFrame(data['quarterlyReports'])
                        df_q['reportType'] = 'quarterly'
                        df = pd.concat([df, df_q])
                    
                    logger.info(f"Balance general obtenido exitosamente para {symbol}")
                    return df
                else:
                    logger.warning(f"No se encontraron datos de balance general para {symbol}")
                    return pd.DataFrame()
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener balance general para {symbol} después de {MAX_RETRIES} intentos.")
                    raise
        
        return pd.DataFrame()
    
    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """
        Obtiene el flujo de caja de la empresa.
        
        Args:
            symbol: Símbolo de la acción
            
        Returns:
            DataFrame con datos del flujo de caja
        """
        if not self.alpha_api_key:
            logger.error("Se requiere API key de Alpha Vantage para flujo de caja")
            raise ValueError("Alpha Vantage API key es necesaria para flujo de caja")
        
        logger.info(f"Obteniendo flujo de caja para {symbol}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit()
                url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={self.alpha_api_key}"
                r = requests.get(url)
                data = r.json()
                
                if "Error Message" in data:
                    raise ValueError(f"Error en Alpha Vantage: {data['Error Message']}")
                
                # Convertir a DataFrame
                if 'annualReports' in data:
                    df = pd.DataFrame(data['annualReports'])
                    df['reportType'] = 'annual'
                    
                    if 'quarterlyReports' in data:
                        df_q = pd.DataFrame(data['quarterlyReports'])
                        df_q['reportType'] = 'quarterly'
                        df = pd.concat([df, df_q])
                    
                    logger.info(f"Flujo de caja obtenido exitosamente para {symbol}")
                    return df
                else:
                    logger.warning(f"No se encontraron datos de flujo de caja para {symbol}")
                    return pd.DataFrame()
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener flujo de caja para {symbol} después de {MAX_RETRIES} intentos.")
                    raise
        
        return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame, expected_columns: List[str]) -> Tuple[bool, str]:
        """
        Valida la calidad y completitud de los datos extraídos.
        
        Args:
            df: DataFrame a validar
            expected_columns: Lista de columnas que debería tener el DataFrame
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if df.empty:
            return False, "DataFrame está vacío"
        
        # Verificar columnas
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            return False, f"Faltan columnas: {missing_cols}"
        
        # Verificar valores nulos
        null_counts = df[expected_columns].isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            return True, f"Advertencia: Valores nulos en columnas: {null_cols.to_dict()}"
        
        return True, "Datos válidos"
    
    def batch_extract_stocks(self, symbols: List[str], output_dir: str) -> Dict[str, str]:
        """
        Extrae datos para múltiples símbolos y los guarda en archivos.
        
        Args:
            symbols: Lista de símbolos a extraer
            output_dir: Directorio donde guardar los datos
            
        Returns:
            Diccionario con resultados por símbolo
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for symbol in symbols:
            try:
                # Obtener datos diarios
                data = self.get_daily_stock_data(symbol)
                valid, message = self.validate_data(
                    data, ['date', 'open', 'high', 'low', 'close', 'volume']
                )
                
                if valid:
                    # Guardar en CSV
                    file_path = os.path.join(output_dir, f"{symbol}_daily.csv")
                    data.to_csv(file_path, index=False)
                    logger.info(f"Datos guardados en {file_path}")
                    
                    results[symbol] = "OK" if "Advertencia" not in message else message
                else:
                    results[symbol] = f"Error: {message}"
                    
            except Exception as e:
                logger.error(f"Error procesando {symbol}: {str(e)}")
                results[symbol] = f"Error: {str(e)}"
        
        # Guardar resumen de resultados
        with open(os.path.join(output_dir, "extraction_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results


if __name__ == "__main__":
    # Ejemplo de uso
    extractor = StockDataExtractor()
    
    # Lista de ejemplo para pruebas
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "^GSPC"]
    
    # Extraer y guardar datos
    results = extractor.batch_extract_stocks(
        symbols=test_symbols,
        output_dir="data/stocks/raw"
    )
    
    print("Resultados de la extracción:")
    for symbol, result in results.items():
        print(f"{symbol}: {result}")