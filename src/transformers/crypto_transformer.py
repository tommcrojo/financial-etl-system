#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para transformación y procesamiento de datos de criptomonedas.
Realiza limpieza, normalización y cálculo de indicadores para criptomonedas.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/crypto_transformer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('crypto_transformer')


class CryptoDataTransformer:
    """
    Clase para transformar y procesar datos de criptomonedas.
    Incluye limpieza, normalización y cálculo de indicadores específicos.
    """
    
    def __init__(self):
        """Inicializa el transformador de datos de criptomonedas."""
        self.scaler = None
    
    def clean_crypto_data(self, df: pd.DataFrame, handle_missing: str = 'interpolate',
                        remove_outliers: bool = True) -> pd.DataFrame:
        """
        Limpia los datos de criptomonedas, maneja valores faltantes y outliers.
        
        Args:
            df: DataFrame con datos de criptomonedas
            handle_missing: Método para manejar valores faltantes ('drop', 'interpolate', 'forward')
            remove_outliers: Si True, detecta y maneja outliers
            
        Returns:
            DataFrame limpio
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para limpiar")
            return df
        
        logger.info(f"Limpiando datos de criptomonedas, shape inicial: {df.shape}")
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        # 1. Convertir fecha a datetime si es necesario
        if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
            try:
                data['date'] = pd.to_datetime(data['date'])
                logger.info("Columna 'date' convertida a datetime")
            except Exception as e:
                logger.error(f"Error convirtiendo fecha: {str(e)}")
        
        # 2. Ordenar por fecha
        if 'date' in data.columns:
            data = data.sort_values('date')
            logger.info("Datos ordenados por fecha")
        
        # 3. Detectar y eliminar duplicados
        if 'date' in data.columns:
            duplicates = data.duplicated(subset=['date'], keep='first')
            if duplicates.any():
                dup_count = duplicates.sum()
                data = data.drop_duplicates(subset=['date'], keep='first')
                logger.info(f"Eliminados {dup_count} registros duplicados")
        
        # 4. Manejar valores faltantes en columnas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if handle_missing == 'drop':
            # Eliminar filas con valores faltantes
            rows_before = len(data)
            data = data.dropna(subset=numeric_cols)
            rows_after = len(data)
            logger.info(f"Eliminadas {rows_before - rows_after} filas con valores faltantes")
            
        elif handle_missing == 'interpolate':
            # Interpolar valores faltantes
            for col in numeric_cols:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    data[col] = data[col].interpolate(method='linear', limit_direction='both')
                    logger.info(f"Interpolados {null_count} valores en columna '{col}'")
        
        elif handle_missing == 'forward':
            # Forward fill
            for col in numeric_cols:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    data[col] = data[col].fillna(method='ffill')
                    # Si quedan nulos al inicio, usar backward fill
                    remaining_nulls = data[col].isnull().sum()
                    if remaining_nulls > 0:
                        data[col] = data[col].fillna(method='bfill')
                    logger.info(f"Aplicado forward fill a {null_count} valores en columna '{col}'")
        
        # 5. Detectar y manejar outliers
        if remove_outliers and len(data) > 30:  # Solo si hay suficientes datos
            # Columnas de precio y volumen son especialmente susceptibles a outliers
            for col in ['price', 'volume', 'market_cap']:
                if col in data.columns:
                    # Para criptomonedas, usamos un enfoque más conservador debido a la volatilidad natural
                    # Calcular z-scores
                    z_scores = stats.zscore(data[col], nan_policy='omit')
                    abs_z_scores = np.abs(z_scores)
                    # Umbral más alto para criptomonedas: 4 sigmas (99.994% intervalo de confianza)
                    outliers_mask = abs_z_scores > 4
                    outliers_count = outliers_mask.sum()
                    
                    if outliers_count > 0:
                        # Para criptomonedas, es mejor no afectar picos legítimos de volatilidad
                        # Solo reemplazar outliers extremos
                        upper_bound = data[col].quantile(0.9975)
                        lower_bound = data[col].quantile(0.0025)
                        
                        # Reemplazar outliers superiores
                        upper_outliers = (data[col] > upper_bound) & outliers_mask
                        data.loc[upper_outliers, col] = upper_bound
                        
                        # Reemplazar outliers inferiores
                        lower_outliers = (data[col] < lower_bound) & outliers_mask
                        data.loc[lower_outliers, col] = lower_bound
                        
                        logger.info(f"Manejados {outliers_count} outliers extremos en columna '{col}'")
        
        # 6. Asegurar valores coherentes
        # Volumen y precio no pueden ser negativos
        for col in ['price', 'volume', 'market_cap']:
            if col in data.columns:
                neg_values = (data[col] < 0).sum()
                if neg_values > 0:
                    data.loc[data[col] < 0, col] = 0
                    logger.info(f"Corregidos {neg_values} valores negativos en columna '{col}'")
        
        logger.info(f"Limpieza completada, shape final: {data.shape}")
        return data
    
    def normalize_crypto_data(self, df: pd.DataFrame, method: str = 'log',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normaliza los datos de criptomonedas utilizando varios métodos.
        
        Args:
            df: DataFrame con datos de criptomonedas
            method: Método de normalización ('minmax', 'standard', 'log', 'percent')
            columns: Lista de columnas a normalizar (None = todas numéricas)
            
        Returns:
            DataFrame con datos normalizados
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para normalizar")
            return df
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        # Seleccionar columnas a normalizar
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            # Excluir columnas de fecha o índice
            columns = [col for col in columns if col not in ['date', 'timestamp']]
        
        logger.info(f"Normalizando columnas {columns} usando método '{method}'")
        
        if method == 'minmax':
            # Normalización Min-Max (escala a rango [0,1])
            self.scaler = MinMaxScaler()
            data[columns] = self.scaler.fit_transform(data[columns])
            
        elif method == 'standard':
            # Estandarización (media 0, desviación estándar 1)
            self.scaler = StandardScaler()
            data[columns] = self.scaler.fit_transform(data[columns])
            
        elif method == 'log':
            # Transformación logarítmica natural (ideal para datos de criptomonedas)
            for col in columns:
                # Asegurar que no haya valores negativos o cero
                if (data[col] <= 0).any():
                    min_val = data[col].min()
                    if min_val <= 0:
                        # Desplazar para que el mínimo sea positivo
                        shift = abs(min_val) + 1e-6
                        data[col] = data[col] + shift
                        logger.info(f"Columna '{col}' desplazada por {shift} para log transform")
                
                # Aplicar transformación logarítmica
                data[col] = np.log(data[col])
            
        elif method == 'percent':
            # Cambio porcentual (útil para returns diarios)
            for col in columns:
                data[f'{col}_pct_change'] = data[col].pct_change() * 100
            
            # Eliminar primera fila con NaN
            data = data.iloc[1:].reset_index(drop=True)
        
        logger.info(f"Normalización completada con método '{method}'")
        return data
    
    def calculate_crypto_indicators(self, df: pd.DataFrame, include_volatility: bool = True,
                                  include_momentum: bool = True) -> pd.DataFrame:
        """
        Calcula indicadores técnicos específicos para criptomonedas.
        
        Args:
            df: DataFrame con datos de criptomonedas (debe incluir price, volume)
            include_volatility: Si True, incluye indicadores de volatilidad
            include_momentum: Si True, incluye indicadores de momentum
            
        Returns:
            DataFrame con indicadores añadidos
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para calcular indicadores")
            return df
        
        # Verificar columnas necesarias
        required_cols = ['price', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Intentar adaptarse a los nombres de columnas disponibles
            if 'close' in df.columns and 'price' in missing_cols:
                df['price'] = df['close']
                missing_cols.remove('price')
                logger.info("Usando columna 'close' como 'price'")
            
            if missing_cols:
                logger.error(f"Faltan columnas necesarias: {missing_cols}")
                return df
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        # Asegurar que los datos están ordenados cronológicamente
        if 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Calculando indicadores para criptomonedas para {len(data)} registros")
        
        # Crear columnas en formato OHLC para usar funciones de TA-Lib
        if 'price' in data.columns and not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Simular OHLC a partir del precio (para criptomonedas a veces solo tenemos precio de cierre)
            data['close'] = data['price']
            
            # Si no tenemos OHLC completo, estimamos los valores
            if 'open' not in data.columns:
                data['open'] = data['price'].shift(1)
                # Para el primer registro
                if pd.isna(data.loc[0, 'open']):
                    data.loc[0, 'open'] = data.loc[0, 'price']
            
            if 'high' not in data.columns:
                # Estimamos high como max(open, close) * (1 + 0.002)
                data['high'] = data[['open', 'close']].max(axis=1) * 1.002
            
            if 'low' not in data.columns:
                # Estimamos low como min(open, close) * (1 - 0.002)
                data['low'] = data[['open', 'close']].min(axis=1) * 0.998
            
            logger.info("Creadas columnas OHLC a partir del precio para cálculo de indicadores")
        
        # 1. Medias Móviles (diferentes períodos para criptomonedas)
        sma_periods = [7, 21, 50, 200]
        for period in sma_periods:
            data[f'sma_{period}'] = ta.trend.sma_indicator(data['close'], window=period)
        
        # Media móvil exponencial para criptomonedas (reacciona más rápido)
        ema_periods = [7, 21, 50, 100]
        for period in ema_periods:
            data[f'ema_{period}'] = ta.trend.ema_indicator(data['close'], window=period)
        
        logger.info(f"Calculadas medias móviles: SMA {sma_periods}, EMA {ema_periods}")
        
        # 2. RSI (útil para detectar sobreventa/sobrecompra en criptomonedas)
        data['rsi_14'] = ta.momentum.rsi(data['close'], window=14)
        
        # 3. MACD (útil para detectar cambios de tendencia)
        # Usamos períodos más cortos para criptomonedas (9,21,9) en lugar de (12,26,9)
        data['macd_line'] = ta.trend.macd(data['close'], window_slow=21, window_fast=9)
        data['macd_signal'] = ta.trend.macd_signal(data['close'], window_slow=21, window_fast=9, window_sign=9)
        data['macd_histogram'] = ta.trend.macd_diff(data['close'], window_slow=21, window_fast=9, window_sign=9)
        
        logger.info("Calculados RSI y MACD")
        
        # 4. Bandas de Bollinger (útiles para detectar volatilidad)
        # Para criptomonedas usamos períodos más cortos (14 en lugar de 20)
        data['bb_upper'] = ta.volatility.bollinger_hband(data['close'], window=14, window_dev=2)
        data['bb_middle'] = ta.volatility.bollinger_mavg(data['close'], window=14)
        data['bb_lower'] = ta.volatility.bollinger_lband(data['close'], window=14, window_dev=2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        logger.info("Calculadas Bandas de Bollinger")
        
        # 5. Indicadores específicos para criptomonedas
        
        # 5.1 Volatilidad
        if include_volatility:
            # Calcular retornos logarítmicos diarios
            data['log_return'] = np.log(data['close'] / data['close'].shift(1))
            
            # Volatilidad móvil (diferentes ventanas)
            volatility_windows = [7, 14, 30, 90]
            for window in volatility_windows:
                data[f'volatility_{window}d'] = data['log_return'].rolling(window=window).std() * np.sqrt(365)
            
            # Volatilidad histórica anualizada
            if len(data) >= 30:
                data['hist_volatility_30d'] = data['log_return'].rolling(window=30).std() * np.sqrt(365)
            
            # Normalized Range (High-Low)/(SMA20)
            if 'high' in data.columns and 'low' in data.columns:
                data['norm_range'] = (data['high'] - data['low']) / data['close']
                data['norm_range_7d_avg'] = data['norm_range'].rolling(window=7).mean()
            
            logger.info(f"Calculados indicadores de volatilidad para ventanas {volatility_windows}")
        
        # 5.2 Momentum y tendencia
        if include_momentum:
            # Rate of Change (ROC)
            roc_periods = [1, 7, 21, 90]
            for period in roc_periods:
                data[f'roc_{period}d'] = ta.momentum.roc(data['close'], window=period)
            
            # Calcular retornos acumulados para diferentes períodos
            return_periods = [7, 14, 30, 90]
            for period in return_periods:
                if len(data) >= period:
                    # Retorno acumulado en los últimos N días
                    data[f'return_{period}d'] = data['close'] / data['close'].shift(period) - 1
            
            # NVT Signal (Network Value to Transactions Signal) - aproximación
            if 'market_cap' in data.columns and 'volume' in data.columns:
                # NVT = Market Cap / Daily Volume
                data['nvt_ratio'] = data['market_cap'] / data['volume']
                data['nvt_signal'] = data['nvt_ratio'].rolling(window=14).mean()
            
            # Market Dominance (si tenemos datos de market_cap)
            if 'market_cap' in data.columns and 'total_market_cap' in data.columns:
                data['market_dominance'] = data['market_cap'] / data['total_market_cap'] * 100
            
            logger.info(f"Calculados indicadores de momentum y tendencia")
        
        # 6. Indicadores basados en volumen
        if 'volume' in data.columns:
            # On Balance Volume (OBV) - útil para confirmar tendencias
            data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
            
            # Volume SMA
            data['volume_sma_14'] = ta.trend.sma_indicator(data['volume'], window=14)
            
            # Volume/SMA ratio para detectar picos de volumen
            data['volume_ratio'] = data['volume'] / data['volume_sma_14']
            
            # Chaikin Money Flow
            if all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                data['cmf_20'] = ta.volume.chaikin_money_flow(
                    data['high'], data['low'], data['close'], data['volume'], window=20
                )
            
            logger.info("Calculados indicadores basados en volumen")
        
        # 7. Calcular señales de tendencia
        
        # Tendencia basada en SMA
        if 'sma_50' in data.columns and 'sma_200' in data.columns:
            # Golden Cross (SMA50 cruza por encima de SMA200)
            data['golden_cross'] = (
                (data['sma_50'] > data['sma_200']) & 
                (data['sma_50'].shift(1) <= data['sma_200'].shift(1))
            ).astype(int)
            
            # Death Cross (SMA50 cruza por debajo de SMA200)
            data['death_cross'] = (
                (data['sma_50'] < data['sma_200']) & 
                (data['sma_50'].shift(1) >= data['sma_200'].shift(1))
            ).astype(int)
            
            # Tendencia general (1 = alcista, -1 = bajista, 0 = neutral)
            data['trend'] = np.where(data['sma_50'] > data['sma_200'], 1, 
                                   np.where(data['sma_50'] < data['sma_200'], -1, 0))
        
        # 8. Eliminar filas iniciales con NaN (debido a ventanas móviles)
        rows_before = len(data)
        max_window = max(sma_periods + ema_periods)
        data = data.iloc[max_window:].reset_index(drop=True)
        rows_after = len(data)
        logger.info(f"Eliminadas {rows_before - rows_after} filas iniciales con valores NaN")
        
        logger.info(f"Cálculo de indicadores para criptomonedas completado. Total de columnas: {len(data.columns)}")
        return data
    
    def add_market_cycle_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade indicadores para identificar ciclos de mercado en criptomonedas.
        
        Args:
            df: DataFrame con indicadores técnicos calculados
            
        Returns:
            DataFrame con indicadores de ciclos de mercado añadidos
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para añadir indicadores de ciclo")
            return df
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        logger.info("Añadiendo indicadores de ciclos de mercado")
        
        # 1. Identificar posibles fondos locales
        if 'close' in data.columns and len(data) > 10:
            # Un fondo local es un punto donde el precio es menor que en los días anteriores y posteriores
            # Utilizamos una ventana de 10 días a cada lado
            window_size = 10
            data['local_bottom'] = 0
            
            for i in range(window_size, len(data) - window_size):
                before_min = data.loc[i-window_size:i-1, 'close'].min()
                after_min = data.loc[i+1:i+window_size, 'close'].min()
                current = data.loc[i, 'close']
                
                if current < before_min and current < after_min:
                    data.loc[i, 'local_bottom'] = 1
            
            logger.info("Identificados posibles fondos locales")
        
        # 2. Identificar posibles techos locales
        if 'close' in data.columns and len(data) > 10:
            # Un techo local es un punto donde el precio es mayor que en los días anteriores y posteriores
            data['local_top'] = 0
            
            for i in range(window_size, len(data) - window_size):
                before_max = data.loc[i-window_size:i-1, 'close'].max()
                after_max = data.loc[i+1:i+window_size, 'close'].max()
                current = data.loc[i, 'close']
                
                if current > before_max and current > after_max:
                    data.loc[i, 'local_top'] = 1
            
            logger.info("Identificados posibles techos locales")
        
        # 3. Calcular distancia desde máximo histórico (ATH)
        if 'close' in data.columns:
            # Calcular máximo histórico móvil
            data['rolling_max'] = data['close'].cummax()
            
            # Distancia desde ATH (en porcentaje)
            data['drawdown_from_ath'] = (data['close'] / data['rolling_max'] - 1) * 100
            
            logger.info("Calculada distancia desde máximo histórico (ATH)")
        
        # 4. Identificar fases de mercado
        if all(col in data.columns for col in ['rsi_14', 'trend', 'drawdown_from_ath']):
            # Fase de Acumulación: tendencia bajista pero RSI comienza a recuperarse
            data['accumulation_phase'] = (
                (data['trend'] == -1) & 
                (data['rsi_14'] > 50) & 
                (data['rsi_14'].shift(7) < 45) &
                (data['drawdown_from_ath'] < -20)  # Al menos 20% abajo del ATH
            ).astype(int)
            
            # Fase de Tendencia Alcista: tendencia alcista y momentum positivo
            data['bullish_phase'] = (
                (data['trend'] == 1) & 
                (data['rsi_14'] > 55) &
                (data['macd_histogram'] > 0)
            ).astype(int)
            
            # Fase de Distribución: cerca de ATH pero momentum comienza a debilitarse
            data['distribution_phase'] = (
                (data['drawdown_from_ath'] > -10) &  # Dentro del 10% del ATH
                (data['rsi_14'] < 70) &
                (data['rsi_14'].shift(7) > 75) &
                (data['macd_histogram'] < data['macd_histogram'].shift(7))
            ).astype(int)
            
            # Fase Bajista: tendencia bajista y momentum negativo
            data['bearish_phase'] = (
                (data['trend'] == -1) & 
                (data['rsi_14'] < 45) &
                (data['macd_histogram'] < 0)
            ).astype(int)
            
            logger.info("Identificadas posibles fases de mercado")
        
        # 5. Calcular indicador para posibles ciclos de burbuja
        if 'volatility_30d' in data.columns and 'return_90d' in data.columns:
            # Posible burbuja: alta volatilidad y retornos extremos
            data['bubble_indicator'] = (
                (data['volatility_30d'] > data['volatility_30d'].quantile(0.85)) & 
                (data['return_90d'] > data['return_90d'].quantile(0.90))
            ).astype(int)
            
            logger.info("Calculado indicador de posible burbuja")
        
        # 6. Añadir etiquetas simples de ciclo
        if 'drawdown_from_ath' in data.columns and 'trend' in data.columns:
            # Crear etiqueta de ciclo: 1=inicio de ciclo, 2=fase alcista, 3=euforia, 4=fase bajista
            conditions = [
                (data['drawdown_from_ath'] < -60) & (data['trend'] == -1),  # Fondo de mercado
                (data['drawdown_from_ath'] < -20) & (data['trend'] == 1),   # Recuperación
                (data['drawdown_from_ath'] > -20) & (data['trend'] == 1),   # Fase alcista
                (data['drawdown_from_ath'] > -10),                          # Cerca de ATH
                (data['drawdown_from_ath'] < -10) & (data['trend'] == -1)   # Fase bajista
            ]
            
            choices = [1, 2, 3, 4, 5]
            data['market_cycle'] = np.select(conditions, choices, default=0)
            
            # Convertir a etiquetas descriptivas
            cycle_labels = {
                1: 'Fondo de Mercado',
                2: 'Recuperación',
                3: 'Fase Alcista',
                4: 'Euforia (Cerca de ATH)',
                5: 'Fase Bajista'
            }
            
            data['market_cycle_label'] = data['market_cycle'].map(cycle_labels)
            
            logger.info("Añadidas etiquetas de ciclo de mercado")
        
        logger.info("Adición de indicadores de ciclos de mercado completada")
        return data
    
    def process_crypto_batch(self, input_dir: str, output_dir: str,
                           include_indicators: bool = True,
                           include_market_cycles: bool = True) -> Dict[str, Dict]:
        """
        Procesa un lote de archivos de datos de criptomonedas.
        
        Args:
            input_dir: Directorio de entrada con datos de criptomonedas
            output_dir: Directorio de salida para datos procesados
            include_indicators: Si True, calcula indicadores técnicos
            include_market_cycles: Si True, añade indicadores de ciclos de mercado
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        try:
            # Obtener datos globales del mercado si existen
            global_market_data = {}
            global_market_path = os.path.join(input_dir, "global_market_data.json")
            if os.path.exists(global_market_path):
                with open(global_market_path, 'r') as f:
                    global_market_data = json.load(f)
                
                # Guardar versión limpia
                output_global_path = os.path.join(output_dir, "global_market_data_clean.json")
                with open(output_global_path, 'w') as f:
                    json.dump(global_market_data, f, indent=2)
                
                logger.info(f"Datos globales de mercado guardados en {output_global_path}")
            
            # Listar subdirectorios (uno por criptomoneda)
            crypto_dirs = [d for d in os.listdir(input_dir) 
                         if os.path.isdir(os.path.join(input_dir, d)) and 
                            d not in ['.git', 'data']]
            
            if not crypto_dirs:
                logger.warning(f"No se encontraron directorios de criptomonedas en {input_dir}")
                return {"error": "No se encontraron datos para procesar"}
            
            logger.info(f"Procesando datos de {len(crypto_dirs)} criptomonedas")
            
            # Obtener market cap total para cálculo de dominancia
            total_market_cap_series = None
            if global_market_data and 'total_market_cap' in global_market_data:
                # Convertir a serie temporal si es posible
                if isinstance(global_market_data['total_market_cap'], dict):
                    # Obtener datos de market cap total por USD
                    if 'usd' in global_market_data['total_market_cap']:
                        total_market_cap = global_market_data['total_market_cap']['usd']
                        
                        # Si es un dato único, crear serie constante
                        if isinstance(total_market_cap, (int, float)):
                            # No podemos crear serie temporal sin fechas
                            pass
                        elif isinstance(total_market_cap, list) and 'date' in global_market_data:
                            # Si tenemos lista de valores y fechas, crear serie temporal
                            dates = global_market_data.get('date', [])
                            if len(dates) == len(total_market_cap):
                                total_market_cap_series = pd.Series(
                                    total_market_cap,
                                    index=pd.to_datetime(dates)
                                )
            
            # Procesar cada criptomoneda
            for crypto_dir in crypto_dirs:
                crypto_symbol = crypto_dir.upper()
                crypto_results = {
                    'market_data': False,
                    'info': False,
                    'network_data': False,
                    'indicators': False,
                    'market_cycles': False
                }
                
                input_crypto_dir = os.path.join(input_dir, crypto_dir)
                output_crypto_dir = os.path.join(output_dir, crypto_dir)
                os.makedirs(output_crypto_dir, exist_ok=True)
                
                try:
                    # 1. Procesar datos de mercado
                    market_file = f"{crypto_dir}_market_data.csv"
                    market_path = os.path.join(input_crypto_dir, market_file)
                    
                    market_df = None
                    if os.path.exists(market_path):
                        market_df = pd.read_csv(market_path)
                        if not market_df.empty:
                            # Limpiar datos
                            market_df_clean = self.clean_crypto_data(
                                market_df, handle_missing='interpolate', remove_outliers=True
                            )
                            
                            # Guardar datos limpios
                            output_market_path = os.path.join(output_crypto_dir, f"{crypto_dir}_market_clean.csv")
                            market_df_clean.to_csv(output_market_path, index=False)
                            logger.info(f"Datos de mercado limpios guardados en {output_market_path}")
                            crypto_results['market_data'] = True
                            
                            # Añadir total_market_cap para cálculo de dominancia
                            if total_market_cap_series is not None and 'date' in market_df_clean.columns:
                                try:
                                    dates = pd.to_datetime(market_df_clean['date'])
                                    market_df_clean['total_market_cap'] = total_market_cap_series.reindex(
                                        dates, method='nearest'
                                    ).values
                                except Exception as e:
                                    logger.warning(f"No se pudo añadir total_market_cap a {crypto_symbol}: {str(e)}")
                            
                            # 2. Calcular indicadores técnicos
                            if include_indicators:
                                market_df_indicators = self.calculate_crypto_indicators(
                                    market_df_clean, include_volatility=True, include_momentum=True
                                )
                                
                                # Guardar datos con indicadores
                                output_indicators_path = os.path.join(
                                    output_crypto_dir, f"{crypto_dir}_indicators.csv"
                                )
                                market_df_indicators.to_csv(output_indicators_path, index=False)
                                logger.info(f"Indicadores técnicos guardados en {output_indicators_path}")
                                crypto_results['indicators'] = True
                                
                                # 3. Añadir indicadores de ciclo de mercado
                                if include_market_cycles:
                                    market_df_cycles = self.add_market_cycle_indicators(market_df_indicators)
                                    
                                    # Guardar datos con ciclos
                                    output_cycles_path = os.path.join(
                                        output_crypto_dir, f"{crypto_dir}_market_cycles.csv"
                                    )
                                    market_df_cycles.to_csv(output_cycles_path, index=False)
                                    logger.info(f"Indicadores de ciclo guardados en {output_cycles_path}")
                                    crypto_results['market_cycles'] = True
                        else:
                            logger.warning(f"Archivo de datos de mercado vacío para {crypto_symbol}")
                    else:
                        logger.warning(f"No se encontró archivo de datos de mercado para {crypto_symbol}")
                    
                    # 4. Procesar información general
                    info_file = f"{crypto_dir}_info.json"
                    info_path = os.path.join(input_crypto_dir, info_file)
                    
                    if os.path.exists(info_path):
                        with open(info_path, 'r') as f:
                            info_data = json.load(f)
                        
                        if info_data:
                            # No necesita mucho procesamiento, solo guardar en formato limpio
                            output_info_path = os.path.join(output_crypto_dir, f"{crypto_dir}_info_clean.json")
                            with open(output_info_path, 'w') as f:
                                json.dump(info_data, f, indent=2)
                            logger.info(f"Información general guardada en {output_info_path}")
                            crypto_results['info'] = True
                        else:
                            logger.warning(f"Archivo de información vacío para {crypto_symbol}")
                    else:
                        logger.warning(f"No se encontró archivo de información para {crypto_symbol}")
                    
                    # 5. Procesar datos de red
                    network_file = f"{crypto_dir}_network_data.json"
                    network_path = os.path.join(input_crypto_dir, network_file)
                    
                    if os.path.exists(network_path):
                        with open(network_path, 'r') as f:
                            network_data = json.load(f)
                        
                        if network_data and len(network_data) > 2:  # Más que solo símbolo y error
                            # No necesita mucho procesamiento, solo guardar en formato limpio
                            output_network_path = os.path.join(output_crypto_dir, f"{crypto_dir}_network_clean.json")
                            with open(output_network_path, 'w') as f:
                                json.dump(network_data, f, indent=2)
                            logger.info(f"Datos de red guardados en {output_network_path}")
                            crypto_results['network_data'] = True
                        else:
                            logger.warning(f"Archivo de datos de red vacío o incompleto para {crypto_symbol}")
                    else:
                        logger.warning(f"No se encontró archivo de datos de red para {crypto_symbol}")
                    
                except Exception as e:
                    logger.error(f"Error procesando {crypto_symbol}: {str(e)}")
                
                # Guardar resultados para esta criptomoneda
                results[crypto_symbol] = crypto_results
            
            # Guardar resumen de resultados
            with open(os.path.join(output_dir, "processing_results.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info("Procesamiento por lotes completado")
            return results
            
        except Exception as e:
            logger.error(f"Error en procesamiento por lotes: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Ejemplo de uso
    transformer = CryptoDataTransformer()
    
    # Procesar datos de criptomonedas
    results = transformer.process_crypto_batch(
        input_dir="data/crypto/raw",
        output_dir="data/crypto/processed",
        include_indicators=True,
        include_market_cycles=True
    )
    
    print("Resultados del procesamiento:")
    for crypto, result in results.items():
        print(f"{crypto}: {result}")