#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo extractor de datos de criptomonedas.
Obtiene datos de mercado y métricas de red para las principales criptomonedas.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

import requests
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/crypto_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('crypto_extractor')

# Cargar variables de entorno
load_dotenv()
CRYPTO_COMPARE_API_KEY = os.getenv('CRYPTO_COMPARE_API_KEY')

# Constantes
MAX_RETRIES = 3
RETRY_DELAY = 60  # Segundos entre reintentos (respeta límites de API)
DEFAULT_CURRENCY = 'usd'


class CryptoDataExtractor:
    """
    Extractor de datos de criptomonedas utilizando CoinGecko como fuente principal
    y CryptoCompare como respaldo.
    """
    
    def __init__(self, crypto_compare_api_key: Optional[str] = None):
        """
        Inicializa el extractor con clave API opcional para CryptoCompare.
        
        Args:
            crypto_compare_api_key: Clave de API de CryptoCompare (opcional si se define en .env)
        """
        self.coingecko = CoinGeckoAPI()
        self.crypto_compare_key = crypto_compare_api_key or CRYPTO_COMPARE_API_KEY
        
        # Inicializar contadores para respetar límites de tasa
        self.last_coingecko_request = datetime.now() - timedelta(seconds=30)
        self.coingecko_requests_count = 0
        
        # Cache de IDs de monedas (coingecko requiere IDs específicos)
        self.coin_id_cache = {}
    
    def _respect_rate_limit_coingecko(self) -> None:
        """
        Asegura que se respeten los límites de tasa de CoinGecko (50 llamadas/minuto).
        """
        now = datetime.now()
        time_diff = (now - self.last_coingecko_request).total_seconds()
        
        # Reiniciar contador después de 1 minuto
        if time_diff >= 60:
            self.coingecko_requests_count = 0
            
        # Si estamos cerca del límite, esperar
        if self.coingecko_requests_count >= 45:  # Margen de seguridad
            sleep_time = max(0, 60 - time_diff)
            logger.info(f"Acercándose al límite de tasa de CoinGecko. Esperando {sleep_time:.2f} segundos...")
            time.sleep(sleep_time + 1)  # +1 para asegurar
            self.coingecko_requests_count = 0
        
        # Añadir pequeña espera entre solicitudes
        if time_diff < 1.0:
            time.sleep(1.0 - time_diff)
        
        self.coingecko_requests_count += 1
        self.last_coingecko_request = datetime.now()
    
    def get_coin_id(self, symbol: str) -> str:
        """
        Obtiene el ID de CoinGecko para un símbolo de criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda (ej. 'btc', 'eth')
            
        Returns:
            ID de la criptomoneda en CoinGecko
        """
        symbol = symbol.lower()
        
        # Verificar si ya está en caché
        if symbol in self.coin_id_cache:
            return self.coin_id_cache[symbol]
        
        # Mapeo directo para las principales
        common_mappings = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'usdt': 'tether',
            'bnb': 'binancecoin',
            'sol': 'solana',
            'xrp': 'ripple',
            'usdc': 'usd-coin',
            'ada': 'cardano',
            'doge': 'dogecoin',
            'dot': 'polkadot'
        }
        
        if symbol in common_mappings:
            self.coin_id_cache[symbol] = common_mappings[symbol]
            return common_mappings[symbol]
        
        # Si no está en el mapeo común, buscar en la lista completa
        try:
            self._respect_rate_limit_coingecko()
            coins_list = self.coingecko.get_coins_list()
            
            # Buscar coincidencia exacta de símbolo
            for coin in coins_list:
                if coin['symbol'].lower() == symbol:
                    self.coin_id_cache[symbol] = coin['id']
                    return coin['id']
            
            logger.warning(f"No se encontró ID para el símbolo {symbol}")
            return ""
        
        except Exception as e:
            logger.error(f"Error obteniendo ID para {symbol}: {str(e)}")
            return ""
    
    def get_coin_market_data(self, symbol: str, vs_currency: str = DEFAULT_CURRENCY, 
                            days: int = 90) -> pd.DataFrame:
        """
        Obtiene datos históricos de mercado para una criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda
            vs_currency: Moneda de cotización (por defecto 'usd')
            days: Número de días de historia a obtener
            
        Returns:
            DataFrame con datos históricos de mercado
        """
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            logger.error(f"No se pudo obtener ID para {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Obteniendo datos de mercado para {symbol} (ID: {coin_id}) últimos {days} días...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit_coingecko()
                # Obtener datos de mercado de CoinGecko
                market_data = self.coingecko.get_coin_market_chart_by_id(
                    id=coin_id,
                    vs_currency=vs_currency,
                    days=days
                )
                
                # Convertir a DataFrame
                prices_data = []
                for timestamp_ms, price in market_data['prices']:
                    date = datetime.fromtimestamp(timestamp_ms/1000)
                    prices_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'timestamp': timestamp_ms,
                        'price': price
                    })
                
                prices_df = pd.DataFrame(prices_data)
                
                # Añadir volumen y marketcap
                if 'market_caps' in market_data and 'total_volumes' in market_data:
                    # Mapeo de timestamps a datos
                    market_caps = {ts: mc for ts, mc in market_data['market_caps']}
                    volumes = {ts: vol for ts, vol in market_data['total_volumes']}
                    
                    # Añadir datos al DataFrame
                    prices_df['market_cap'] = prices_df['timestamp'].map(
                        lambda ts: market_caps.get(ts, np.nan)
                    )
                    prices_df['volume'] = prices_df['timestamp'].map(
                        lambda ts: volumes.get(ts, np.nan)
                    )
                
                logger.info(f"Datos de mercado obtenidos exitosamente para {symbol}")
                return prices_df
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener datos de mercado para {symbol} después de {MAX_RETRIES} intentos.")
                    
                    # Intentar con CryptoCompare como respaldo
                    if self.crypto_compare_key:
                        logger.info(f"Intentando obtener datos desde CryptoCompare para {symbol}...")
                        return self._get_crypto_compare_historical(symbol, vs_currency, days)
                    else:
                        logger.error("No se pudo usar respaldo de CryptoCompare (falta API key)")
                        raise
        
        return pd.DataFrame()
    
    def _get_crypto_compare_historical(self, symbol: str, vs_currency: str = DEFAULT_CURRENCY,
                                      days: int = 90) -> pd.DataFrame:
        """
        Obtiene datos históricos de CryptoCompare como respaldo.
        
        Args:
            symbol: Símbolo de la criptomoneda
            vs_currency: Moneda de cotización
            days: Número de días de historia
            
        Returns:
            DataFrame con datos históricos
        """
        if not self.crypto_compare_key:
            logger.error("Se requiere API key de CryptoCompare")
            return pd.DataFrame()
        
        symbol = symbol.upper()
        vs_currency = vs_currency.upper()
        
        # Determinar el endpoint según días solicitados
        if days <= 30:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={vs_currency}&limit={days}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={vs_currency}&limit={days}&allData=true"
        
        for attempt in range(MAX_RETRIES):
            try:
                headers = {
                    'authorization': f'Apikey {self.crypto_compare_key}'
                }
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if data['Response'] != 'Success':
                    raise ValueError(f"Error en CryptoCompare: {data.get('Message', 'Unknown error')}")
                
                # Convertir a DataFrame
                df = pd.DataFrame(data['Data']['Data'])
                
                # Convertir timestamps a fechas
                df['date'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%d')
                
                # Renombrar columnas para consistencia
                df.rename(columns={
                    'time': 'timestamp',
                    'close': 'price',
                    'volumefrom': 'volume',
                    'volumeto': 'volume_to'
                }, inplace=True)
                
                logger.info(f"Datos obtenidos exitosamente de CryptoCompare para {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido en CryptoCompare: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener datos de CryptoCompare para {symbol}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_coin_info(self, symbol: str) -> Dict:
        """
        Obtiene información general sobre una criptomoneda.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Diccionario con información de la criptomoneda
        """
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            logger.error(f"No se pudo obtener ID para {symbol}")
            return {}
        
        logger.info(f"Obteniendo información para {symbol} (ID: {coin_id})...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit_coingecko()
                # Obtener datos completos
                coin_data = self.coingecko.get_coin_by_id(
                    id=coin_id,
                    localization='false',
                    tickers=False,
                    market_data=True,
                    community_data=False,
                    developer_data=False,
                    sparkline=False
                )
                
                # Filtrar y simplificar la información
                info = {
                    'id': coin_data.get('id', ''),
                    'symbol': coin_data.get('symbol', '').upper(),
                    'name': coin_data.get('name', ''),
                    'categories': coin_data.get('categories', []),
                    'description': coin_data.get('description', {}).get('en', ''),
                    'market_cap_rank': coin_data.get('market_cap_rank'),
                    'hashing_algorithm': coin_data.get('hashing_algorithm'),
                    'genesis_date': coin_data.get('genesis_date'),
                }
                
                # Añadir datos de mercado si están disponibles
                if 'market_data' in coin_data:
                    market_data = coin_data['market_data']
                    info['current_price'] = market_data.get('current_price', {}).get(DEFAULT_CURRENCY)
                    info['market_cap'] = market_data.get('market_cap', {}).get(DEFAULT_CURRENCY)
                    info['total_volume'] = market_data.get('total_volume', {}).get(DEFAULT_CURRENCY)
                    info['circulating_supply'] = market_data.get('circulating_supply')
                    info['total_supply'] = market_data.get('total_supply')
                    info['max_supply'] = market_data.get('max_supply')
                    
                    # Cambios de precio
                    info['price_change_24h'] = market_data.get('price_change_percentage_24h')
                    info['price_change_7d'] = market_data.get('price_change_percentage_7d')
                    info['price_change_30d'] = market_data.get('price_change_percentage_30d')
                
                logger.info(f"Información obtenida exitosamente para {symbol}")
                return info
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener información para {symbol} después de {MAX_RETRIES} intentos.")
                    raise
        
        return {}
    
    def get_coin_network_data(self, symbol: str) -> Dict:
        """
        Obtiene métricas de red para una criptomoneda.
        Nota: No todas las criptomonedas tienen datos de red disponibles.
        
        Args:
            symbol: Símbolo de la criptomoneda
            
        Returns:
            Diccionario con métricas de red
        """
        # CoinGecko no ofrece directamente métricas de red completas en su API gratuita
        # Para datos más detallados, sería necesario usar otras fuentes como Glassnode o CryptoQuant
        # Esta implementación es simplificada y puede expandirse
        
        try:
            # Obtener datos básicos que pueden incluir algunas métricas de red
            coin_info = self.get_coin_info(symbol)
            
            network_data = {
                'symbol': symbol.upper(),
                'hashing_algorithm': coin_info.get('hashing_algorithm'),
                'genesis_date': coin_info.get('genesis_date'),
                'circulating_supply': coin_info.get('circulating_supply'),
                'total_supply': coin_info.get('total_supply'),
                'max_supply': coin_info.get('max_supply'),
            }
            
            # Si tenemos una API key de CryptoCompare, podemos obtener algunos datos adicionales
            if self.crypto_compare_key:
                try:
                    url = f"https://min-api.cryptocompare.com/data/blockchain/latest?fsym={symbol.upper()}"
                    headers = {
                        'authorization': f'Apikey {self.crypto_compare_key}'
                    }
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data['Response'] == 'Success' and 'Data' in data:
                        blockchain_data = data['Data']
                        
                        # Añadir métricas de red disponibles
                        network_data.update({
                            'active_addresses': blockchain_data.get('active_addresses'),
                            'transaction_count': blockchain_data.get('transaction_count'),
                            'average_transaction_value': blockchain_data.get('average_transaction_value'),
                            'block_height': blockchain_data.get('block_height'),
                            'hashrate': blockchain_data.get('hashrate'),
                            'difficulty': blockchain_data.get('difficulty')
                        })
                except Exception as e:
                    logger.warning(f"No se pudieron obtener datos de blockchain para {symbol}: {str(e)}")
            
            return network_data
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de red para {symbol}: {str(e)}")
            return {
                'symbol': symbol.upper(),
                'error': str(e)
            }
    
    def get_global_market_data(self) -> Dict:
        """
        Obtiene datos globales del mercado de criptomonedas.
        
        Returns:
            Diccionario con datos globales del mercado
        """
        logger.info("Obteniendo datos globales del mercado de criptomonedas...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit_coingecko()
                global_data = self.coingecko.get_global()
                
                logger.info("Datos globales obtenidos exitosamente")
                return global_data
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener datos globales después de {MAX_RETRIES} intentos.")
                    raise
        
        return {}
    
    def get_top_coins(self, limit: int = 100) -> pd.DataFrame:
        """
        Obtiene lista de las principales criptomonedas por capitalización de mercado.
        
        Args:
            limit: Número máximo de monedas a obtener
            
        Returns:
            DataFrame con lista de principales criptomonedas
        """
        logger.info(f"Obteniendo lista de top {limit} criptomonedas...")
        
        for attempt in range(MAX_RETRIES):
            try:
                self._respect_rate_limit_coingecko()
                coins = self.coingecko.get_coins_markets(
                    vs_currency=DEFAULT_CURRENCY,
                    order='market_cap_desc',
                    per_page=limit,
                    page=1,
                    sparkline=False,
                    price_change_percentage='24h,7d,30d'
                )
                
                # Convertir a DataFrame
                df = pd.DataFrame(coins)
                
                logger.info(f"Lista de top {len(df)} criptomonedas obtenida exitosamente")
                return df
                
            except Exception as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Error al obtener lista de criptomonedas después de {MAX_RETRIES} intentos.")
                    raise
        
        return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
        """
        Valida la calidad y completitud de los datos extraídos.
        
        Args:
            df: DataFrame a validar
            required_columns: Lista de columnas requeridas
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if df.empty:
            return False, "DataFrame está vacío"
        
        # Verificar columnas requeridas
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Faltan columnas requeridas: {missing_cols}"
        
        # Verificar si hay datos
        if len(df) < 5:
            return False, f"Muy pocos registros: {len(df)}"
        
        # Verificar valores nulos en columnas críticas
        null_counts = df[required_columns].isnull().sum()
        critical_nulls = null_counts[null_counts > len(df) * 0.5]  # Más del 50% nulos
        
        if not critical_nulls.empty:
            return False, f"Demasiados valores nulos en columnas críticas: {critical_nulls.to_dict()}"
        
        return True, "Datos válidos"
    
    def extract_crypto_batch(self, symbols: List[str], days: int = 90,
                            output_dir: str = "data/crypto/raw") -> Dict[str, Dict]:
        """
        Extrae datos para múltiples criptomonedas y los guarda.
        
        Args:
            symbols: Lista de símbolos a extraer
            days: Número de días de historia a obtener
            output_dir: Directorio donde guardar los datos
            
        Returns:
            Diccionario con resultados por símbolo
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # 1. Obtener y guardar datos globales de mercado
        try:
            global_data = self.get_global_market_data()
            if global_data:
                global_path = os.path.join(output_dir, "global_market_data.json")
                with open(global_path, 'w') as f:
                    json.dump(global_data, f, indent=2)
                logger.info(f"Datos globales guardados en {global_path}")
        except Exception as e:
            logger.error(f"Error obteniendo datos globales: {str(e)}")
        
        # 2. Obtener y guardar lista de top monedas
        try:
            top_coins = self.get_top_coins(limit=100)
            if not top_coins.empty:
                top_path = os.path.join(output_dir, "top_coins.csv")
                top_coins.to_csv(top_path, index=False)
                logger.info(f"Lista de top monedas guardada en {top_path}")
        except Exception as e:
            logger.error(f"Error obteniendo lista de top monedas: {str(e)}")
        
        # 3. Procesar cada símbolo
        for symbol in symbols:
            symbol_results = {
                'market_data': False,
                'coin_info': False,
                'network_data': False
            }
            
            symbol_dir = os.path.join(output_dir, symbol.lower())
            os.makedirs(symbol_dir, exist_ok=True)
            
            try:
                # a. Datos de mercado
                market_df = self.get_coin_market_data(symbol, days=days)
                valid, message = self.validate_data(
                    market_df, ['date', 'price', 'volume']
                )
                
                if valid:
                    market_path = os.path.join(symbol_dir, f"{symbol.lower()}_market_data.csv")
                    market_df.to_csv(market_path, index=False)
                    logger.info(f"Datos de mercado guardados en {market_path}")
                    symbol_results['market_data'] = True
                else:
                    logger.warning(f"Datos de mercado para {symbol} no válidos: {message}")
                
                # b. Información general
                coin_info = self.get_coin_info(symbol)
                if coin_info:
                    info_path = os.path.join(symbol_dir, f"{symbol.lower()}_info.json")
                    with open(info_path, 'w') as f:
                        json.dump(coin_info, f, indent=2)
                    logger.info(f"Información general guardada en {info_path}")
                    symbol_results['coin_info'] = True
                else:
                    logger.warning(f"No se obtuvo información general para {symbol}")
                
                # c. Datos de red
                network_data = self.get_coin_network_data(symbol)
                if network_data and len(network_data) > 3:  # Más que solo símbolo y error
                    network_path = os.path.join(symbol_dir, f"{symbol.lower()}_network_data.json")
                    with open(network_path, 'w') as f:
                        json.dump(network_data, f, indent=2)
                    logger.info(f"Datos de red guardados en {network_path}")
                    symbol_results['network_data'] = True
                else:
                    logger.warning(f"Datos de red no disponibles o insuficientes para {symbol}")
                
            except Exception as e:
                logger.error(f"Error procesando {symbol}: {str(e)}")
            
            # Guardar resultado para este símbolo
            results[symbol] = symbol_results
        
        # Guardar resumen de resultados
        with open(os.path.join(output_dir, "extraction_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results


if __name__ == "__main__":
    # Ejemplo de uso
    extractor = CryptoDataExtractor()
    
    # Lista de ejemplo para pruebas
    test_symbols = ["BTC", "ETH", "SOL", "ADA", "DOGE"]
    
    # Extraer y guardar datos
    results = extractor.extract_crypto_batch(
        symbols=test_symbols,
        days=30,
        output_dir="data/crypto/raw"
    )
    
    print("Resultados de la extracción:")
    for symbol, result in results.items():
        print(f"{symbol}: {result}")