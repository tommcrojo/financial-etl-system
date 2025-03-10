#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para transformación y procesamiento de datos bursátiles.
Realiza limpieza, normalización y cálculo de indicadores técnicos.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
import ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/stock_transformer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stock_transformer')


class StockDataTransformer:
    """
    Clase para transformar y procesar datos de acciones e índices.
    Incluye limpieza, normalización y cálculo de indicadores técnicos.
    """
    
    def __init__(self):
        """Inicializa el transformador de datos bursátiles."""
        self.scaler = None
    
    def clean_stock_data(self, df: pd.DataFrame, handle_missing: str = 'interpolate',
                        remove_outliers: bool = True) -> pd.DataFrame:
        """
        Limpia los datos de acciones, maneja valores faltantes y outliers.
        
        Args:
            df: DataFrame con datos de acciones
            handle_missing: Método para manejar valores faltantes ('drop', 'interpolate', 'forward')
            remove_outliers: Si True, detecta y maneja outliers
            
        Returns:
            DataFrame limpio
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para limpiar")
            return df
        
        logger.info(f"Limpiando datos bursátiles, shape inicial: {df.shape}")
        
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
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    # Calcular z-scores
                    z_scores = stats.zscore(data[col], nan_policy='omit')
                    abs_z_scores = np.abs(z_scores)
                    outliers_mask = abs_z_scores > 3  # Z-score > 3 (99.7% intervalo de confianza)
                    outliers_count = outliers_mask.sum()
                    
                    if outliers_count > 0:
                        # Reemplazar outliers con el valor del percentil 95 o 5
                        upper_bound = data[col].quantile(0.95)
                        lower_bound = data[col].quantile(0.05)
                        
                        # Reemplazar outliers superiores
                        upper_outliers = (data[col] > upper_bound) & outliers_mask
                        data.loc[upper_outliers, col] = upper_bound
                        
                        # Reemplazar outliers inferiores
                        lower_outliers = (data[col] < lower_bound) & outliers_mask
                        data.loc[lower_outliers, col] = lower_bound
                        
                        logger.info(f"Manejados {outliers_count} outliers en columna '{col}'")
        
        # 6. Asegurar valores coherentes
        # Volumen no puede ser negativo
        if 'volume' in data.columns:
            neg_volume = (data['volume'] < 0).sum()
            if neg_volume > 0:
                data.loc[data['volume'] < 0, 'volume'] = 0
                logger.info(f"Corregidos {neg_volume} valores negativos de volumen")
        
        # High debe ser >= que low
        if 'high' in data.columns and 'low' in data.columns:
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                # Intercambiar high y low donde high < low
                idx = data['high'] < data['low']
                data.loc[idx, ['high', 'low']] = data.loc[idx, ['low', 'high']].values
                logger.info(f"Corregidos {invalid_hl} casos donde high < low")
        
        logger.info(f"Limpieza completada, shape final: {data.shape}")
        return data
    
    def normalize_stock_data(self, df: pd.DataFrame, method: str = 'minmax',
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normaliza los datos de acciones utilizando varios métodos.
        
        Args:
            df: DataFrame con datos de acciones
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
            # Transformación logarítmica natural (para datos sesgados)
            for col in columns:
                # Asegurar que no haya valores negativos o cero
                if (data[col] <= 0).any():
                    min_val = data[col].min()
                    if min_val <= 0:
                        # Desplazar para que el mínimo sea positivo
                        shift = abs(min_val) + 1
                        data[col] = data[col] + shift
                        logger.info(f"Columna '{col}' desplazada por {shift} para log transform")
                
                # Aplicar transformación logarítmica
                data[col] = np.log(data[col])
            
        elif method == 'percent':
            # Cambio porcentual
            for col in columns:
                data[f'{col}_pct_change'] = data[col].pct_change() * 100
            
            # Eliminar primera fila con NaN
            data = data.iloc[1:].reset_index(drop=True)
        
        logger.info(f"Normalización completada con método '{method}'")
        return data
    
    def calculate_technical_indicators(self, df: pd.DataFrame, include_all: bool = False,
                                     custom_periods: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
        """
        Calcula indicadores técnicos para datos de acciones.
        
        Args:
            df: DataFrame con datos de acciones (debe incluir OHLCV)
            include_all: Si True, incluye todos los indicadores disponibles
            custom_periods: Diccionario de períodos personalizados por indicador
            
        Returns:
            DataFrame con indicadores técnicos añadidos
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para calcular indicadores")
            return df
        
        # Verificar columnas necesarias
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Faltan columnas necesarias: {missing_cols}")
            return df
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        # Asegurar que los datos están ordenados cronológicamente
        if 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Calculando indicadores técnicos para {len(data)} registros")
        
        # Definir períodos por defecto
        default_periods = {
            'sma': [20, 50, 200],
            'ema': [12, 26, 50],
            'bb': [20],
            'rsi': [14],
            'macd': [(12, 26, 9)],  # (rápido, lento, señal)
            'stoch': [(14, 3)],  # (k_periodo, d_periodo)
            'adx': [14],
            'atr': [14]
        }
        
        # Usar períodos personalizados si se proporcionan
        periods = custom_periods if custom_periods else default_periods
        
        # Calcular indicadores según configuración
        
        # 1. Medias Móviles Simples (SMA)
        for period in periods.get('sma', default_periods['sma']):
            data[f'sma_{period}'] = ta.trend.sma_indicator(data['close'], window=period)
            logger.info(f"Calculada SMA de período {period}")
        
        # 2. Medias Móviles Exponenciales (EMA)
        for period in periods.get('ema', default_periods['ema']):
            data[f'ema_{period}'] = ta.trend.ema_indicator(data['close'], window=period)
            logger.info(f"Calculada EMA de período {period}")
        
        # 3. Bandas de Bollinger
        for period in periods.get('bb', default_periods['bb']):
            # Banda superior
            data[f'bb_high_{period}'] = ta.volatility.bollinger_hband(
                data['close'], window=period, window_dev=2
            )
            # Banda media (SMA)
            data[f'bb_mid_{period}'] = ta.volatility.bollinger_mavg(
                data['close'], window=period
            )
            # Banda inferior
            data[f'bb_low_{period}'] = ta.volatility.bollinger_lband(
                data['close'], window=period, window_dev=2
            )
            # Indicador %B (posición relativa dentro de las bandas)
            data[f'bb_pct_b_{period}'] = ta.volatility.bollinger_pband(
                data['close'], window=period, window_dev=2
            )
            logger.info(f"Calculadas Bandas de Bollinger de período {period}")
        
        # 4. RSI (Relative Strength Index)
        for period in periods.get('rsi', default_periods['rsi']):
            data[f'rsi_{period}'] = ta.momentum.rsi(data['close'], window=period)
            logger.info(f"Calculado RSI de período {period}")
        
        # 5. MACD (Moving Average Convergence Divergence)
        for fast, slow, signal in periods.get('macd', default_periods['macd']):
            # MACD Line
            data[f'macd_{fast}_{slow}_{signal}'] = ta.trend.macd(
                data['close'], window_slow=slow, window_fast=fast
            )
            # Signal Line
            data[f'macd_signal_{fast}_{slow}_{signal}'] = ta.trend.macd_signal(
                data['close'], window_slow=slow, window_fast=fast, window_sign=signal
            )
            # Histogram
            data[f'macd_hist_{fast}_{slow}_{signal}'] = ta.trend.macd_diff(
                data['close'], window_slow=slow, window_fast=fast, window_sign=signal
            )
            logger.info(f"Calculado MACD ({fast},{slow},{signal})")
        
        # 6. Estocástico
        for k_period, d_period in periods.get('stoch', default_periods['stoch']):
            # %K
            data[f'stoch_k_{k_period}'] = ta.momentum.stoch(
                data['high'], data['low'], data['close'], window=k_period, smooth_window=d_period
            )
            # %D (Media móvil de %K)
            data[f'stoch_d_{k_period}_{d_period}'] = ta.momentum.stoch_signal(
                data['high'], data['low'], data['close'], window=k_period, smooth_window=d_period
            )
            logger.info(f"Calculado Estocástico (K:{k_period}, D:{d_period})")
        
        # 7. ADX (Average Directional Index)
        for period in periods.get('adx', default_periods['adx']):
            data[f'adx_{period}'] = ta.trend.adx(
                data['high'], data['low'], data['close'], window=period
            )
            data[f'adx_pos_{period}'] = ta.trend.adx_pos(
                data['high'], data['low'], data['close'], window=period
            )
            data[f'adx_neg_{period}'] = ta.trend.adx_neg(
                data['high'], data['low'], data['close'], window=period
            )
            logger.info(f"Calculado ADX de período {period}")
        
        # 8. ATR (Average True Range)
        for period in periods.get('atr', default_periods['atr']):
            data[f'atr_{period}'] = ta.volatility.average_true_range(
                data['high'], data['low'], data['close'], window=period
            )
            logger.info(f"Calculado ATR de período {period}")
        
        # Indicadores adicionales si include_all es True
        if include_all:
            # 9. OBV (On-Balance Volume)
            data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
            logger.info("Calculado OBV")
            
            # 10. CCI (Commodity Channel Index)
            data['cci_20'] = ta.trend.cci(
                data['high'], data['low'], data['close'], window=20
            )
            logger.info("Calculado CCI de período 20")
            
            # 11. Williams %R
            data['williams_r_14'] = ta.momentum.williams_r(
                data['high'], data['low'], data['close'], lbp=14
            )
            logger.info("Calculado Williams %R de período 14")
            
            # 12. Rate of Change (ROC)
            data['roc_10'] = ta.momentum.roc(data['close'], window=10)
            logger.info("Calculado ROC de período 10")
            
            # 13. Money Flow Index (MFI)
            data['mfi_14'] = ta.volume.money_flow_index(
                data['high'], data['low'], data['close'], data['volume'], window=14
            )
            logger.info("Calculado MFI de período 14")
            
            # 14. Ichimoku Cloud
            data['ichimoku_a'] = ta.trend.ichimoku_a(
                data['high'], data['low'], window1=9, window2=26
            )
            data['ichimoku_b'] = ta.trend.ichimoku_b(
                data['high'], data['low'], window2=26, window3=52
            )
            logger.info("Calculado Ichimoku Cloud")
        
        # Calcular rendimientos diarios y acumulados
        data['daily_return'] = data['close'].pct_change() * 100
        data['cumulative_return'] = (1 + data['daily_return'] / 100).cumprod() - 1
        data['cumulative_return'] = data['cumulative_return'] * 100  # Convertir a porcentaje
        logger.info("Calculados rendimientos diarios y acumulados")
        
        # Calcular volatilidad móvil (desviación estándar de rendimientos)
        data['volatility_20d'] = data['daily_return'].rolling(window=20).std()
        logger.info("Calculada volatilidad móvil de 20 días")
        
        # Eliminar filas iniciales con NaN (debido a indicadores que requieren datos históricos)
        rows_before = len(data)
        data = data.iloc[max(periods.get('sma', default_periods['sma'])):].reset_index(drop=True)
        rows_after = len(data)
        logger.info(f"Eliminadas {rows_before - rows_after} filas iniciales con valores NaN")
        
        logger.info(f"Cálculo de indicadores técnicos completo. Total de columnas: {len(data.columns)}")
        return data
    
    def add_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade señales de tendencia basadas en indicadores técnicos.
        
        Args:
            df: DataFrame con indicadores técnicos calculados
            
        Returns:
            DataFrame con señales de tendencia añadidas
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para añadir señales")
            return df
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        logger.info("Añadiendo señales de tendencia basadas en indicadores técnicos")
        
        # 1. Señal de cruce de Medias Móviles (Golden Cross / Death Cross)
        if 'sma_50' in data.columns and 'sma_200' in data.columns:
            # Golden Cross (SMA50 cruza por encima de SMA200): señal alcista
            data['golden_cross'] = (
                (data['sma_50'] > data['sma_200']) & 
                (data['sma_50'].shift(1) <= data['sma_200'].shift(1))
            ).astype(int)
            
            # Death Cross (SMA50 cruza por debajo de SMA200): señal bajista
            data['death_cross'] = (
                (data['sma_50'] < data['sma_200']) & 
                (data['sma_50'].shift(1) >= data['sma_200'].shift(1))
            ).astype(int)
            
            logger.info("Añadidas señales de Golden Cross y Death Cross")
        
        # 2. Señal de cruce MACD
        if 'macd_12_26_9' in data.columns and 'macd_signal_12_26_9' in data.columns:
            # Cruce alcista (MACD cruza por encima de su señal)
            data['macd_bull_cross'] = (
                (data['macd_12_26_9'] > data['macd_signal_12_26_9']) & 
                (data['macd_12_26_9'].shift(1) <= data['macd_signal_12_26_9'].shift(1))
            ).astype(int)
            
            # Cruce bajista (MACD cruza por debajo de su señal)
            data['macd_bear_cross'] = (
                (data['macd_12_26_9'] < data['macd_signal_12_26_9']) & 
                (data['macd_12_26_9'].shift(1) >= data['macd_signal_12_26_9'].shift(1))
            ).astype(int)
            
            logger.info("Añadidas señales de cruce MACD")
        
        # 3. Señales de RSI (sobrecompra/sobreventa)
        if 'rsi_14' in data.columns:
            # Sobreventa (RSI < 30): posible señal de compra
            data['rsi_oversold'] = (data['rsi_14'] < 30).astype(int)
            
            # Sobrecompra (RSI > 70): posible señal de venta
            data['rsi_overbought'] = (data['rsi_14'] > 70).astype(int)
            
            # Salida de sobreventa (RSI cruza por encima de 30)
            data['rsi_bull_cross'] = (
                (data['rsi_14'] > 30) & 
                (data['rsi_14'].shift(1) <= 30)
            ).astype(int)
            
            # Salida de sobrecompra (RSI cruza por debajo de 70)
            data['rsi_bear_cross'] = (
                (data['rsi_14'] < 70) & 
                (data['rsi_14'].shift(1) >= 70)
            ).astype(int)
            
            logger.info("Añadidas señales de RSI")
        
        # 4. Señales de Bandas de Bollinger
        if all(col in data.columns for col in ['close', 'bb_high_20', 'bb_low_20']):
            # Precio fuera de banda superior (posible sobrecompra)
            data['bb_upper_break'] = (data['close'] > data['bb_high_20']).astype(int)
            
            # Precio fuera de banda inferior (posible sobreventa)
            data['bb_lower_break'] = (data['close'] < data['bb_low_20']).astype(int)
            
            # Retorno a banda desde arriba (posible señal de venta)
            data['bb_upper_return'] = (
                (data['close'] < data['bb_high_20']) & 
                (data['close'].shift(1) >= data['bb_high_20'].shift(1))
            ).astype(int)
            
            # Retorno a banda desde abajo (posible señal de compra)
            data['bb_lower_return'] = (
                (data['close'] > data['bb_low_20']) & 
                (data['close'].shift(1) <= data['bb_low_20'].shift(1))
            ).astype(int)
            
            logger.info("Añadidas señales de Bandas de Bollinger")
        
        # 5. Señales de Estocástico
        if 'stoch_k_14' in data.columns and 'stoch_d_14_3' in data.columns:
            # Cruce alcista (K cruza por encima de D)
            data['stoch_bull_cross'] = (
                (data['stoch_k_14'] > data['stoch_d_14_3']) & 
                (data['stoch_k_14'].shift(1) <= data['stoch_d_14_3'].shift(1))
            ).astype(int)
            
            # Cruce bajista (K cruza por debajo de D)
            data['stoch_bear_cross'] = (
                (data['stoch_k_14'] < data['stoch_d_14_3']) & 
                (data['stoch_k_14'].shift(1) >= data['stoch_d_14_3'].shift(1))
            ).astype(int)
            
            # Sobreventa (K y D < 20)
            data['stoch_oversold'] = (
                (data['stoch_k_14'] < 20) & 
                (data['stoch_d_14_3'] < 20)
            ).astype(int)
            
            # Sobrecompra (K y D > 80)
            data['stoch_overbought'] = (
                (data['stoch_k_14'] > 80) & 
                (data['stoch_d_14_3'] > 80)
            ).astype(int)
            
            logger.info("Añadidas señales de Estocástico")
        
        # 6. Tendencia según ADX
        if 'adx_14' in data.columns:
            # Tendencia fuerte (ADX > 25)
            data['strong_trend'] = (data['adx_14'] > 25).astype(int)
            
            # Tendencia muy fuerte (ADX > 50)
            data['very_strong_trend'] = (data['adx_14'] > 50).astype(int)
            
            # Tendencia débil (ADX < 20)
            data['weak_trend'] = (data['adx_14'] < 20).astype(int)
            
            logger.info("Añadidas señales de tendencia basadas en ADX")
        
        # 7. Señal combinada (peso ponderado de múltiples señales)
        signal_columns = [
            'macd_bull_cross', 'macd_bear_cross',
            'rsi_bull_cross', 'rsi_bear_cross',
            'bb_upper_return', 'bb_lower_return',
            'stoch_bull_cross', 'stoch_bear_cross'
        ]
        
        available_signals = [col for col in signal_columns if col in data.columns]
        
        if available_signals:
            # Clasificar señales como alcistas (1) o bajistas (-1)
            bullish_signals = [
                'macd_bull_cross', 'rsi_bull_cross', 
                'bb_lower_return', 'stoch_bull_cross'
            ]
            
            bearish_signals = [
                'macd_bear_cross', 'rsi_bear_cross', 
                'bb_upper_return', 'stoch_bear_cross'
            ]
            
            # Inicializar la señal combinada
            data['combined_signal'] = 0
            
            # Sumar señales alcistas
            for signal in bullish_signals:
                if signal in data.columns:
                    data['combined_signal'] += data[signal]
            
            # Restar señales bajistas
            for signal in bearish_signals:
                if signal in data.columns:
                    data['combined_signal'] -= data[signal]
            
            logger.info("Añadida señal combinada basada en múltiples indicadores")
        
        # 8. Etiqueta de tendencia general
        if 'close' in data.columns:
            # Calcular tendencia basada en media móvil de 20 días
            if 'sma_20' not in data.columns:
                data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
            
            # Tendencia alcista (precio por encima de SMA20 y SMA20 en ascenso)
            data['uptrend'] = (
                (data['close'] > data['sma_20']) & 
                (data['sma_20'] > data['sma_20'].shift(1))
            ).astype(int)
            
            # Tendencia bajista (precio por debajo de SMA20 y SMA20 en descenso)
            data['downtrend'] = (
                (data['close'] < data['sma_20']) & 
                (data['sma_20'] < data['sma_20'].shift(1))
            ).astype(int)
            
            # Etiqueta de tendencia (-1: bajista, 0: neutral, 1: alcista)
            data['trend_label'] = data['uptrend'] - data['downtrend']
            
            logger.info("Añadidas etiquetas de tendencia general")
        
        logger.info("Adición de señales de tendencia completada")
        return data
    
    def process_stock_batch(self, input_dir: str, output_dir: str,
                           include_indicators: bool = True,
                           include_signals: bool = True) -> Dict[str, str]:
        """
        Procesa un lote de archivos de datos bursátiles.
        
        Args:
            input_dir: Directorio de entrada con archivos CSV brutos
            output_dir: Directorio de salida para archivos procesados
            include_indicators: Si True, calcula indicadores técnicos
            include_signals: Si True, añade señales de tendencia
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        try:
            # Listar archivos CSV en el directorio de entrada
            csv_files = [f for f in os.listdir(input_dir) if f.endswith('_daily.csv')]
            
            if not csv_files:
                logger.warning(f"No se encontraron archivos '_daily.csv' en {input_dir}")
                return {"error": "No se encontraron archivos para procesar"}
            
            logger.info(f"Procesando {len(csv_files)} archivos de datos bursátiles")
            
            for csv_file in csv_files:
                try:
                    symbol = csv_file.split('_')[0]
                    input_path = os.path.join(input_dir, csv_file)
                    
                    # Cargar datos
                    df = pd.read_csv(input_path)
                    
                    if df.empty:
                        logger.warning(f"Archivo vacío: {csv_file}")
                        results[symbol] = "Archivo vacío"
                        continue
                    
                    # 1. Limpiar datos
                    df_clean = self.clean_stock_data(df, handle_missing='interpolate', remove_outliers=True)
                    
                    # 2. Calcular indicadores técnicos
                    if include_indicators:
                        df_indicators = self.calculate_technical_indicators(df_clean)
                    else:
                        df_indicators = df_clean
                    
                    # 3. Añadir señales de tendencia
                    if include_signals and include_indicators:
                        df_processed = self.add_trend_signals(df_indicators)
                    else:
                        df_processed = df_indicators
                    
                    # Guardar resultados
                    output_path = os.path.join(output_dir, f"{symbol}_processed.csv")
                    df_processed.to_csv(output_path, index=False)
                    
                    logger.info(f"Datos procesados guardados en {output_path}")
                    results[symbol] = "Procesado exitosamente"
                    
                except Exception as e:
                    logger.error(f"Error procesando {csv_file}: {str(e)}")
                    results[csv_file] = f"Error: {str(e)}"
            
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
    transformer = StockDataTransformer()
    
    # Procesar todos los archivos en el directorio
    results = transformer.process_stock_batch(
        input_dir="data/stocks/raw",
        output_dir="data/stocks/processed",
        include_indicators=True,
        include_signals=True
    )
    
    print("Resultados del procesamiento:")
    for symbol, result in results.items():
        print(f"{symbol}: {result}")