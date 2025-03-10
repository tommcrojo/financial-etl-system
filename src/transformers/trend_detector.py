#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para detección y análisis de tendencias en datos financieros.
Identifica patrones, ciclos y señales de cambio de tendencia.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats, signal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.cluster import KMeans
import pandas_ta as ta

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trend_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('trend_detector')


class TrendDetector:
    """
    Clase para detectar y analizar tendencias en datos financieros.
    """
    
    def __init__(self):
        """Inicializa el detector de tendencias."""
        pass
    
    def load_asset_data(self, data_dir: str, asset_type: str, symbol: str,
                       use_processed: bool = True) -> pd.DataFrame:
        """
        Carga datos de un activo específico.
        
        Args:
            data_dir: Directorio base de datos
            asset_type: Tipo de activo ('stocks', 'crypto', 'indices')
            symbol: Símbolo del activo
            use_processed: Si True, utiliza datos procesados; si False, datos brutos
            
        Returns:
            DataFrame con datos del activo
        """
        symbol = symbol.lower().replace('^', '')  # Manejar índices como ^GSPC
        
        # Determinar ruta según tipo de activo y si usamos datos procesados
        if asset_type == 'stocks' or asset_type == 'indices':
            subfolder = f"stocks/{'processed' if use_processed else 'raw'}"
            if use_processed:
                file_suffix = '_processed.csv'
            else:
                file_suffix = '_daily.csv'
        elif asset_type == 'crypto':
            subfolder = f"crypto/{'processed' if use_processed else 'raw'}"
            if use_processed:
                # Para criptos, intentar el archivo más completo primero
                file_suffix = '_market_cycles.csv'
            else:
                file_suffix = '_market_data.csv'
        else:
            logger.error(f"Tipo de activo no válido: {asset_type}")
            return pd.DataFrame()
        
        # Construir ruta completa
        file_path = os.path.join(data_dir, subfolder, f"{symbol}{file_suffix}")
        
        # Para criptos, intentar archivos alternativos si el principal no existe
        if asset_type == 'crypto' and use_processed and not os.path.exists(file_path):
            alternatives = [
                f"{symbol}_indicators.csv",
                f"{symbol}_market_clean.csv"
            ]
            
            for alt_suffix in alternatives:
                alt_path = os.path.join(data_dir, subfolder, alt_suffix)
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
        
        # Cargar datos
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # Convertir fecha a datetime si existe
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                logger.info(f"Datos cargados exitosamente para {symbol} desde {file_path}")
                return df
            except Exception as e:
                logger.error(f"Error cargando datos para {symbol}: {str(e)}")
                return pd.DataFrame()
        else:
            logger.warning(f"Archivo no encontrado: {file_path}")
            return pd.DataFrame()
    
    def check_stationarity(self, series: pd.Series, window: int = 30) -> Dict:
        """
        Evalúa la estacionariedad de una serie temporal mediante pruebas estadísticas.
        
        Args:
            series: Serie temporal a evaluar
            window: Tamaño de ventana para rolling statistics
            
        Returns:
            Diccionario con resultados de pruebas de estacionariedad
        """
        # Eliminar valores nulos
        series = series.dropna()
        
        if len(series) < window * 2:
            logger.warning(f"Serie demasiado corta para análisis de estacionariedad. Se requieren al menos {window*2} puntos.")
            return {'error': 'Serie demasiado corta'}
        
        logger.info("Evaluando estacionariedad de serie temporal")
        
        results = {}
        
        # Calcular estadísticas móviles
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Prueba Dickey-Fuller Aumentada (ADF)
        # H0: La serie tiene raíz unitaria (no es estacionaria)
        # H1: La serie no tiene raíz unitaria (es estacionaria)
        try:
            adf_result = adfuller(series.values, autolag='AIC')
            results['adf_test'] = {
                'test_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            logger.error(f"Error en prueba ADF: {str(e)}")
            results['adf_test'] = {'error': str(e)}
        
        # Prueba KPSS
        # H0: La serie es estacionaria
        # H1: La serie no es estacionaria
        try:
            kpss_result = kpss(series.values, regression='c')
            results['kpss_test'] = {
                'test_statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        except Exception as e:
            logger.error(f"Error en prueba KPSS: {str(e)}")
            results['kpss_test'] = {'error': str(e)}
        
        # Consenso de estacionariedad
        if ('adf_test' in results and 'is_stationary' in results['adf_test'] and
            'kpss_test' in results and 'is_stationary' in results['kpss_test']):
            
            adf_stationary = results['adf_test']['is_stationary']
            kpss_stationary = results['kpss_test']['is_stationary']
            
            if adf_stationary and kpss_stationary:
                results['consensus'] = "Fuerte evidencia de estacionariedad"
            elif adf_stationary and not kpss_stationary:
                results['consensus'] = "Evidencia mixta: posible tendencia"
            elif not adf_stationary and kpss_stationary:
                results['consensus'] = "Evidencia mixta: posible estacionariedad con drift"
            else:
                results['consensus'] = "Fuerte evidencia de no estacionariedad (serie con tendencia)"
        
        # Calcular estadísticas descriptivas
        results['rolling_stats'] = {
            'mean_min': rolling_mean.min(),
            'mean_max': rolling_mean.max(),
            'std_min': rolling_std.min(),
            'std_max': rolling_std.max(),
            'mean_change_pct': (rolling_mean.iloc[-1] / rolling_mean.iloc[window]) - 1 if len(rolling_mean) > window else None,
            'std_change_pct': (rolling_std.iloc[-1] / rolling_std.iloc[window]) - 1 if len(rolling_std) > window else None
        }
        
        logger.info("Análisis de estacionariedad completado")
        return results
    
    def decompose_series(self, series: pd.Series, period: int = None,
                       model: str = 'additive') -> Dict[str, pd.Series]:
        """
        Descompone una serie temporal en componentes de tendencia, estacionalidad y residuos.
        
        Args:
            series: Serie temporal a descomponer
            period: Período de estacionalidad (None para detección automática)
            model: Tipo de modelo ('additive' o 'multiplicative')
            
        Returns:
            Diccionario con componentes de la serie
        """
        # Eliminar valores nulos
        series = series.dropna()
        
        # Detectar período automáticamente si no se especifica
        if period is None:
            # Usar autocorrelación para detectar estacionalidad
            try:
                # Calcular autocorrelación hasta 1/3 de la longitud de la serie
                max_lag = min(int(len(series) / 3), 365)
                acf = [series.autocorr(lag=i) for i in range(1, max_lag + 1)]
                
                # Buscar picos en la autocorrelación
                peaks, _ = signal.find_peaks(acf, height=0.1, distance=5)
                
                if len(peaks) > 0:
                    # Usar el primer pico significativo como período
                    period = peaks[0] + 1
                    logger.info(f"Período detectado automáticamente: {period}")
                else:
                    # Valor por defecto si no se detecta estacionalidad
                    period = 30  # Mensual para datos financieros diarios
                    logger.info(f"No se detectó estacionalidad clara, usando período por defecto: {period}")
            except Exception as e:
                logger.error(f"Error detectando período: {str(e)}")
                period = 30
        
        # Verificar que hay suficientes datos
        if len(series) < period * 2:
            logger.warning(f"Serie demasiado corta para descomposición con período {period}")
            return {'error': 'Serie demasiado corta para el período especificado'}
        
        logger.info(f"Descomponiendo serie temporal con período {period}, modelo {model}")
        
        try:
            # Realizar descomposición
            decomposition = seasonal_decompose(
                series, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            result = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': model
            }
            
            logger.info("Descomposición completada exitosamente")
            return result
            
        except Exception as e:
            logger.error(f"Error en descomposición: {str(e)}")
            return {'error': str(e)}
    
    def identify_trend_direction(self, series: pd.Series, 
                               window: int = 20,
                               smooth: bool = True) -> Dict:
        """
        Identifica la dirección y fuerza de la tendencia.
        
        Args:
            series: Serie temporal a analizar
            window: Tamaño de ventana para cálculos
            smooth: Si True, suaviza la serie antes del análisis
            
        Returns:
            Diccionario con información sobre la tendencia
        """
        # Eliminar valores nulos
        series = series.dropna()
        
        if len(series) < window * 2:
            logger.warning(f"Serie demasiado corta para análisis de tendencia")
            return {'error': 'Serie demasiado corta'}
        
        # Opcional: suavizar la serie con media móvil
        if smooth:
            series_smooth = series.rolling(window=int(window/2)).mean().dropna()
        else:
            series_smooth = series
        
        logger.info(f"Analizando dirección de tendencia con ventana {window}")
        
        # Calcular inclinación (pendiente) para diferentes ventanas
        slopes = {}
        windows = [window, window*2, window*4]  # Corto, medio y largo plazo
        
        for w in windows:
            if len(series_smooth) < w:
                continue
                
            # Calcular pendiente por regresión lineal
            x = np.array(range(w))
            y = series_smooth.iloc[-w:].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Normalizar pendiente (cambio porcentual por período)
            norm_slope = slope / series_smooth.iloc[-w] * 100
            
            slopes[f'{w}_periods'] = {
                'slope': slope,
                'normalized_slope': norm_slope,
                'r_squared': r_value**2,
                'direction': 'uptrend' if slope > 0 else 'downtrend',
                'strength': abs(r_value)  # Fuerza de la tendencia basada en R
            }
        
        # Calcular indicadores de tendencia
        # 1. ADX (Average Directional Index) si tenemos datos OHLC
        adx_value = None
        if all(col in series.index for col in ['high', 'low', 'close']):
            try:
                high = series.loc['high']
                low = series.loc['low']
                close = series.loc['close']
                adx = ta.trend.ADXIndicator(high, low, close, window=14)
                adx_value = adx.adx().iloc[-1]
            except:
                pass
        
        # 2. Clasificación de tendencia basada en EMAs (9, 21, 50)
        ema_trend = None
        if len(series) > 50:
            ema9 = ta.trend.ema_indicator(series, window=9).iloc[-1]
            ema21 = ta.trend.ema_indicator(series, window=21).iloc[-1]
            ema50 = ta.trend.ema_indicator(series, window=50).iloc[-1]
            
            if ema9 > ema21 > ema50:
                ema_trend = "strong_uptrend"
            elif ema9 < ema21 < ema50:
                ema_trend = "strong_downtrend"
            elif ema9 > ema21 and ema21 < ema50:
                ema_trend = "potential_reversal_up"
            elif ema9 < ema21 and ema21 > ema50:
                ema_trend = "potential_reversal_down"
            else:
                ema_trend = "mixed_sideways"
        
        # 3. Calcular porcentaje actual desde máximo/mínimo
        if len(series) > window:
            current = series.iloc[-1]
            max_value = series.iloc[-window:].max()
            min_value = series.iloc[-window:].min()
            
            pct_from_max = (current / max_value - 1) * 100
            pct_from_min = (current / min_value - 1) * 100
        else:
            pct_from_max = None
            pct_from_min = None
        
        result = {
            'slopes': slopes,
            'current_trend': slopes[f'{window}_periods']['direction'] if f'{window}_periods' in slopes else None,
            'long_term_trend': slopes[f'{window*4}_periods']['direction'] if f'{window*4}_periods' in slopes else None,
            'adx_value': adx_value,
            'ema_trend': ema_trend,
            'percent_from_period_max': pct_from_max,
            'percent_from_period_min': pct_from_min,
            'current_value': series.iloc[-1],
            'start_value': series.iloc[0],
            'overall_change_pct': (series.iloc[-1] / series.iloc[0] - 1) * 100,
            'volatility': series.pct_change().std() * 100  # Volatilidad como desviación estándar de cambio porcentual
        }
        
        logger.info("Análisis de dirección de tendencia completado")
        return result
    
    def detect_trend_changes(self, series: pd.Series, window: int = 20, 
                           threshold: float = 0.15) -> List[Dict]:
        """
        Detecta cambios significativos en la tendencia.
        
        Args:
            series: Serie temporal a analizar
            window: Tamaño de ventana para detectar cambios
            threshold: Umbral de cambio para considerar un cambio de tendencia
            
        Returns:
            Lista de diccionarios con fechas y detalles de cambios detectados
        """
        # Asegurar que tenemos un índice temporal
        has_date_index = False
        original_index = None
        
        if not isinstance(series.index, pd.DatetimeIndex):
            if hasattr(series, 'date') and isinstance(series.date, pd.Series):
                # Si tenemos una columna de fecha, usar como índice
                original_index = series.index
                series = series.set_index('date')
                has_date_index = True
            elif hasattr(series, 'index') and 'date' in series.index:
                # Si 'date' está en el índice (MultiIndex)
                has_date_index = True
        else:
            has_date_index = True
        
        # Eliminar valores nulos
        series = series.dropna()
        
        if len(series) < window * 3:
            logger.warning(f"Serie demasiado corta para detección de cambios de tendencia")
            return []
        
        logger.info(f"Detectando cambios de tendencia con ventana {window}, umbral {threshold}")
        
        # Calcular pendientes móviles
        slopes = []
        dates = []
        
        for i in range(len(series) - window*2 + 1):
            # Ventana 1: período anterior
            window1 = series.iloc[i:i+window]
            # Ventana 2: período actual
            window2 = series.iloc[i+window:i+window*2]
            
            # Calcular pendientes
            x1 = np.array(range(len(window1)))
            y1 = window1.values
            x2 = np.array(range(len(window2)))
            y2 = window2.values
            
            slope1, _, r_value1, _, _ = stats.linregress(x1, y1)
            slope2, _, r_value2, _, _ = stats.linregress(x2, y2)
            
            # Normalizar pendientes (como porcentaje del valor medio)
            norm_slope1 = slope1 / window1.mean() if window1.mean() != 0 else 0
            norm_slope2 = slope2 / window2.mean() if window2.mean() != 0 else 0
            
            # Guardar resultado
            slopes.append({
                'slope1': norm_slope1,
                'slope2': norm_slope2,
                'r1': r_value1,
                'r2': r_value2,
                'change': norm_slope2 - norm_slope1,
                'relative_change': abs(norm_slope2 - norm_slope1) / (abs(norm_slope1) + 1e-10),
                'index': i + window  # Punto medio entre ventanas
            })
            
            # Guardar fecha si está disponible
            if has_date_index:
                dates.append(series.index[i+window])
        
        # Convertir a DataFrame para facilitar análisis
        slopes_df = pd.DataFrame(slopes)
        
        # Detectar cambios significativos
        changes = []
        
        for i in range(1, len(slopes_df)):
            current = slopes_df.iloc[i]
            prev = slopes_df.iloc[i-1]
            
            # Condiciones para cambio significativo:
            # 1. Cambio de signo en la pendiente (cambio de dirección)
            sign_change = (current['slope2'] * prev['slope2']) < 0
            
            # 2. Cambio relativo grande en la magnitud de la pendiente
            magnitude_change = current['relative_change'] > threshold
            
            # 3. Ambas regresiones tienen buen ajuste (R² alto)
            good_fit = (current['r1']**2 > 0.6) and (current['r2']**2 > 0.6)
            
            if (sign_change or magnitude_change) and good_fit:
                change_point = {
                    'index': current['index'],
                    'type': 'reversal' if sign_change else 'acceleration',
                    'magnitude': current['relative_change'],
                    'before_slope': prev['slope2'],
                    'after_slope': current['slope2'],
                    'value': series.iloc[int(current['index'])],
                }
                
                # Añadir fecha si está disponible
                if has_date_index and len(dates) > i:
                    change_point['date'] = dates[i]
                
                changes.append(change_point)
        
        # Filtrar cambios demasiado cercanos (mantener solo el más significativo)
        if len(changes) > 1:
            filtered_changes = [changes[0]]
            min_distance = window / 2
            
            for change in changes[1:]:
                # Calcular distancia al último cambio añadido
                last_change = filtered_changes[-1]
                distance = change['index'] - last_change['index']
                
                if distance > min_distance:
                    # Suficientemente lejos, añadir
                    filtered_changes.append(change)
                else:
                    # Demasiado cerca, conservar el más significativo
                    if change['magnitude'] > last_change['magnitude']:
                        filtered_changes[-1] = change
            
            changes = filtered_changes
        
        # Clasificar cambios (tendencia alcista/bajista)
        for change in changes:
            if change['before_slope'] < 0 and change['after_slope'] > 0:
                change['trend_change'] = 'bullish_reversal'
            elif change['before_slope'] > 0 and change['after_slope'] < 0:
                change['trend_change'] = 'bearish_reversal'
            elif change['after_slope'] > change['before_slope'] and change['after_slope'] > 0:
                change['trend_change'] = 'bullish_acceleration'
            elif change['after_slope'] < change['before_slope'] and change['after_slope'] < 0:
                change['trend_change'] = 'bearish_acceleration'
            elif change['after_slope'] > change['before_slope'] and change['after_slope'] < 0:
                change['trend_change'] = 'bearish_deceleration'
            elif change['after_slope'] < change['before_slope'] and change['after_slope'] > 0:
                change['trend_change'] = 'bullish_deceleration'
            else:
                change['trend_change'] = 'unknown'
        
        logger.info(f"Detectados {len(changes)} cambios de tendencia significativos")
        
        # Restaurar índice original si es necesario
        if original_index is not None:
            series.index = original_index
        
        return changes
    
    def identify_support_resistance(self, series: pd.Series, window: int = 20,
                                  noise_reduction: float = 0.05,
                                  n_levels: int = 3) -> Dict:
        """
        Identifica niveles de soporte y resistencia mediante análisis de histograma y clustering.
        
        Args:
            series: Serie temporal con precios
            window: Tamaño de ventana para análisis
            noise_reduction: Factor para reducción de ruido (0-1)
            n_levels: Número de niveles a identificar
            
        Returns:
            Diccionario con niveles de soporte y resistencia identificados
        """
        # Eliminar valores nulos
        series = series.dropna()
        
        if len(series) < window * 2:
            logger.warning(f"Serie demasiado corta para análisis de soporte/resistencia")
            return {'error': 'Serie demasiado corta'}
        
        logger.info(f"Identificando niveles de soporte y resistencia")
        
        # Para análisis más reciente, enfocarse en la última parte de la serie
        recent_data = series.iloc[-window*10:] if len(series) > window*10 else series
        current_price = series.iloc[-1]
        
        # Método 1: Identificar niveles mediante análisis de histograma
        hist_levels = {}
        
        # Calcular histograma
        hist, bin_edges = np.histogram(recent_data, bins=min(50, len(recent_data)//5))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Identificar picos en el histograma (áreas de concentración de precios)
        peaks, _ = signal.find_peaks(hist, height=np.max(hist)*noise_reduction, distance=3)
        
        if len(peaks) > 0:
            # Ordenar picos por altura (mayor concentración de precios)
            peak_heights = hist[peaks]
            sorted_indices = np.argsort(peak_heights)[::-1]
            
            # Tomar los n_levels principales picos o todos si hay menos
            top_peaks = [peaks[i] for i in sorted_indices[:min(n_levels, len(peaks))]]
            
            # Extraer niveles de precio correspondientes
            price_levels = [bin_centers[peak] for peak in top_peaks]
            
            # Clasificar como soporte o resistencia basado en precio actual
            supports = [level for level in price_levels if level < current_price]
            resistances = [level for level in price_levels if level > current_price]
            
            hist_levels = {
                'supports': sorted(supports, reverse=True),
                'resistances': sorted(resistances),
                'strongest_level': bin_centers[peaks[sorted_indices[0]]]
            }
        
        # Método 2: Identificar niveles mediante clustering de máximos/mínimos locales
        peaks_method = {}
        
        # Identificar máximos y mínimos locales
        local_max = []
        local_min = []
        
        for i in range(window, len(recent_data) - window):
            if recent_data.iloc[i] == max(recent_data.iloc[i-window:i+window]):
                local_max.append(recent_data.iloc[i])
            elif recent_data.iloc[i] == min(recent_data.iloc[i-window:i+window]):
                local_min.append(recent_data.iloc[i])
        
        # Aplicar clustering si hay suficientes puntos
        if len(local_max) >= n_levels and len(local_min) >= n_levels:
            try:
                # Clustering para resistencias (máximos locales)
                kmeans_max = KMeans(n_clusters=min(n_levels, len(local_max)), random_state=42)
                kmeans_max.fit(np.array(local_max).reshape(-1, 1))
                resistance_levels = sorted(kmeans_max.cluster_centers_.flatten())
                
                # Clustering para soportes (mínimos locales)
                kmeans_min = KMeans(n_clusters=min(n_levels, len(local_min)), random_state=42)
                kmeans_min.fit(np.array(local_min).reshape(-1, 1))
                support_levels = sorted(kmeans_min.cluster_centers_.flatten(), reverse=True)
                
                peaks_method = {
                    'supports': support_levels,
                    'resistances': resistance_levels
                }
                
            except Exception as e:
                logger.error(f"Error en clustering para soporte/resistencia: {str(e)}")
        
        # Combinar resultados de ambos métodos
        combined_levels = {
            'histogram_method': hist_levels,
            'peaks_method': peaks_method,
            'current_price': current_price
        }
        
        # Calcular distancia a niveles (para evaluar proximidad)
        all_supports = (
            hist_levels.get('supports', []) + 
            peaks_method.get('supports', [])
        )
        all_resistances = (
            hist_levels.get('resistances', []) + 
            peaks_method.get('resistances', [])
        )
        
        if all_supports:
            # Encontrar soporte más cercano por arriba
            closest_support = max([s for s in all_supports if s < current_price], default=None)
            if closest_support:
                combined_levels['closest_support'] = closest_support
                combined_levels['distance_to_support'] = (current_price / closest_support - 1) * 100
        
        if all_resistances:
            # Encontrar resistencia más cercana por arriba
            closest_resistance = min([r for r in all_resistances if r > current_price], default=None)
            if closest_resistance:
                combined_levels['closest_resistance'] = closest_resistance
                combined_levels['distance_to_resistance'] = (closest_resistance / current_price - 1) * 100
        
        logger.info("Análisis de soporte/resistencia completado")
        return combined_levels
    
    def detect_seasonality(self, series: pd.Series, max_lag: int = 365) -> Dict:
        """
        Detecta patrones estacionales en la serie temporal.
        
        Args:
            series: Serie temporal a analizar
            max_lag: Máximo retraso a considerar en la autocorrelación
            
        Returns:
            Diccionario con información sobre estacionalidad detectada
        """
        # Eliminar valores nulos
        series = series.dropna()
        
        # Asegurar que hay suficientes datos
        if len(series) < max_lag * 1.5:
            logger.warning(f"Serie demasiado corta para análisis de estacionalidad con max_lag={max_lag}")
            max_lag = int(len(series) / 3)
            logger.info(f"Ajustando max_lag a {max_lag}")
        
        if len(series) < 30:
            logger.warning("Serie demasiado corta para análisis de estacionalidad")
            return {'error': 'Serie demasiado corta'}
        
        logger.info(f"Analizando estacionalidad con max_lag={max_lag}")
        
        # Calcular rendimientos para normalizar
        returns = series.pct_change().dropna()
        
        # Calcular autocorrelación
        acf = [returns.autocorr(lag=i) for i in range(1, min(max_lag + 1, len(returns) // 3))]
        
        # Detectar picos significativos
        peaks, properties = signal.find_peaks(acf, height=0.1, distance=5)
        
        # Verificar significancia estadística
        significant_periods = []
        
        if len(peaks) > 0:
            # Para cada pico, calculamos intervalo de confianza al 95%
            confidence_level = 1.96 / np.sqrt(len(returns))
            
            for peak in peaks:
                period = peak + 1  # +1 porque los índices de peaks empiezan en 0
                correlation = acf[peak]
                
                if abs(correlation) > confidence_level:
                    significant_periods.append({
                        'period': period,
                        'correlation': correlation,
                        'significance': abs(correlation) / confidence_level
                    })
        
        # Ordenar por significancia
        significant_periods = sorted(significant_periods, key=lambda x: x['significance'], reverse=True)
        
        # Analizar patrones específicos en datos financieros
        weekday_pattern = None
        monthly_pattern = None
        quarterly_pattern = None
        
        # Intentar detectar patrón semanal si tenemos un índice de fechas
        if (isinstance(series.index, pd.DatetimeIndex) or 
            (hasattr(series, 'date') and isinstance(series.date, pd.Series))):
            
            # Obtener fechas
            if hasattr(series, 'date'):
                dates = pd.to_datetime(series.date)
            else:
                dates = series.index
            
            # Calcular rendimientos por día de la semana
            returns_with_dates = pd.Series(returns.values, index=dates[1:])
            weekday_returns = returns_with_dates.groupby(returns_with_dates.index.dayofweek).mean()
            
            # Verificar si hay patrón significativo
            if weekday_returns.max() - weekday_returns.min() > 0.001:
                weekday_pattern = {
                    'returns_by_day': weekday_returns.to_dict(),
                    'best_day': weekday_returns.idxmax(),
                    'worst_day': weekday_returns.idxmin(),
                    'significance': (weekday_returns.max() - weekday_returns.min()) / weekday_returns.std()
                }
            
            # Calcular rendimientos por mes
            month_returns = returns_with_dates.groupby(returns_with_dates.index.month).mean()
            
            if month_returns.max() - month_returns.min() > 0.01:
                monthly_pattern = {
                    'returns_by_month': month_returns.to_dict(),
                    'best_month': month_returns.idxmax(),
                    'worst_month': month_returns.idxmin(),
                    'significance': (month_returns.max() - month_returns.min()) / month_returns.std()
                }
            
            # Calcular rendimientos por trimestre
            quarter_returns = returns_with_dates.groupby(returns_with_dates.index.quarter).mean()
            
            if quarter_returns.max() - quarter_returns.min() > 0.01:
                quarterly_pattern = {
                    'returns_by_quarter': quarter_returns.to_dict(),
                    'best_quarter': quarter_returns.idxmax(),
                    'worst_quarter': quarter_returns.idxmin(),
                    'significance': (quarter_returns.max() - quarter_returns.min()) / quarter_returns.std()
                }
        
        result = {
            'has_significant_seasonality': len(significant_periods) > 0,
            'periods': significant_periods,
            'weekday_pattern': weekday_pattern,
            'monthly_pattern': monthly_pattern,
            'quarterly_pattern': quarterly_pattern,
            'confidence_threshold': confidence_level if 'confidence_level' in locals() else None
        }
        
        logger.info(f"Análisis de estacionalidad completado: {len(significant_periods)} períodos significativos encontrados")
        return result
    
    def analyze_market_phases(self, series: pd.Series, window: int = 90) -> Dict:
        """
        Identifica fases de mercado (acumulación, tendencia, distribución, etc.).
        
        Args:
            series: Serie temporal con precios
            window: Tamaño de ventana para análisis
            
        Returns:
            Diccionario con información sobre la fase de mercado actual
        """
        # Eliminar valores nulos
        series = series.dropna()
        
        if len(series) < window * 1.5:
            logger.warning(f"Serie demasiado corta para análisis de fases de mercado")
            return {'error': 'Serie demasiado corta'}
        
        logger.info(f"Analizando fases de mercado con ventana {window}")
        
        try:
            # Calcular indicadores clave
            # 1. ATH (All Time High) y distancia desde ATH
            ath = series.max()
            current_price = series.iloc[-1]
            drawdown_from_ath = (current_price / ath - 1) * 100
            
            # 2. Volatilidad reciente vs. histórica
            recent_volatility = series.iloc[-window:].pct_change().std() * 100 * np.sqrt(252)  # Anualizada
            historical_volatility = series.pct_change().std() * 100 * np.sqrt(252)  # Anualizada
            volatility_ratio = recent_volatility / historical_volatility if historical_volatility != 0 else 1
            
            # 3. Rendimiento reciente vs. tendencia de largo plazo
            recent_return = (series.iloc[-1] / series.iloc[-window] - 1) * 100
            long_term_slope = np.polyfit(range(len(series)), series.values, 1)[0]
            normalized_long_term_slope = long_term_slope / series.mean() * 100
            
            # 4. Momentum reciente
            if len(series) >= 14:
                rsi = ta.momentum.rsi(series, window=14).iloc[-1]
            else:
                rsi = None
            
            # Detectar fase de mercado basada en los indicadores
            # Definir umbrales para clasificación
            # Estos son valores empíricos y podrían ajustarse según el tipo de activo
            phase = None
            confidence = 0
            
            # Fase de Acumulación
            if (drawdown_from_ath < -20 and 
                recent_return > 0 and 
                volatility_ratio < 0.8 and 
                rsi is not None and 40 < rsi < 60):
                phase = "accumulation"
                confidence = 0.7
            
            # Fase de Tendencia Alcista
            elif (recent_return > 10 and 
                 normalized_long_term_slope > 0 and 
                 volatility_ratio < 1.2 and 
                 rsi is not None and rsi > 50):
                phase = "bullish_trend"
                confidence = 0.8
            
            # Fase de Distribución
            elif (drawdown_from_ath > -10 and 
                 recent_return < 5 and 
                 volatility_ratio > 1.2 and 
                 rsi is not None and rsi > 70):
                phase = "distribution"
                confidence = 0.6
            
            # Fase de Tendencia Bajista
            elif (recent_return < -10 and 
                 normalized_long_term_slope < 0 and 
                 volatility_ratio > 1 and 
                 rsi is not None and rsi < 50):
                phase = "bearish_trend"
                confidence = 0.8
            
            # Fase de Capitulación
            elif (drawdown_from_ath < -30 and 
                 recent_return < -20 and 
                 volatility_ratio > 1.5 and 
                 rsi is not None and rsi < 30):
                phase = "capitulation"
                confidence = 0.7
            
            # Fase Lateral (Rango)
            elif (abs(recent_return) < 5 and 
                 abs(normalized_long_term_slope) < 2 and 
                 volatility_ratio < 0.8):
                phase = "sideways"
                confidence = 0.6
            
            else:
                phase = "indeterminate"
                confidence = 0.3
            
            result = {
                'current_phase': phase,
                'confidence': confidence,
                'indicators': {
                    'price': current_price,
                    'ath': ath,
                    'drawdown_from_ath': drawdown_from_ath,
                    'recent_volatility': recent_volatility,
                    'historical_volatility': historical_volatility,
                    'volatility_ratio': volatility_ratio,
                    'recent_return': recent_return,
                    'long_term_slope': normalized_long_term_slope,
                    'rsi': rsi
                }
            }
            
            logger.info(f"Análisis de fases de mercado completado: {phase} (confianza: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis de fases de mercado: {str(e)}")
            return {'error': str(e)}
    
    def run_trend_analysis(self, data_dir: str, asset_type: str, symbol: str,
                          output_dir: str = None) -> Dict:
        """
        Ejecuta un análisis completo de tendencias para un activo.
        
        Args:
            data_dir: Directorio con datos procesados
            asset_type: Tipo de activo ('stocks', 'crypto', 'indices')
            symbol: Símbolo del activo
            output_dir: Directorio para guardar resultados (opcional)
            
        Returns:
            Diccionario con resultados del análisis
        """
        # Cargar datos
        df = self.load_asset_data(data_dir, asset_type, symbol)
        
        if df.empty:
            logger.error(f"No se pudieron cargar datos para {symbol}")
            return {'error': f"No se pudieron cargar datos para {symbol}"}
        
        # Determinar columna de precio
        price_col = None
        if asset_type == 'crypto':
            price_col = 'price' if 'price' in df.columns else 'close'
        else:
            price_col = 'close' if 'close' in df.columns else 'adjusted_close'
        
        if price_col not in df.columns:
            logger.error(f"Columna de precio no encontrada en datos de {symbol}")
            return {'error': f"Columna de precio no encontrada en datos de {symbol}"}
        
        logger.info(f"Iniciando análisis completo de tendencias para {symbol} ({asset_type})")
        
        # Serie de precios
        price_series = df[price_col]
        
        # Resultados
        results = {
            'symbol': symbol,
            'asset_type': asset_type,
            'data_points': len(df),
            'first_date': df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else None,
            'last_date': df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else None,
        }
        
        # 1. Análisis de dirección de tendencia
        try:
            trend_direction = self.identify_trend_direction(
                price_series, window=min(20, len(price_series)//4)
            )
            results['trend_direction'] = trend_direction
        except Exception as e:
            logger.error(f"Error en análisis de dirección de tendencia: {str(e)}")
            results['trend_direction'] = {'error': str(e)}
        
        # 2. Detección de cambios de tendencia
        try:
            trend_changes = self.detect_trend_changes(
                price_series, window=min(20, len(price_series)//4)
            )
            # Convertir objetos numpy a python nativos para JSON
            for change in trend_changes:
                if 'date' in change and hasattr(change['date'], 'strftime'):
                    change['date'] = change['date'].strftime('%Y-%m-%d')
                
                for key, val in change.items():
                    if hasattr(val, 'item'):  # Convertir tipos numpy
                        change[key] = val.item()
            
            results['trend_changes'] = trend_changes
        except Exception as e:
            logger.error(f"Error en detección de cambios de tendencia: {str(e)}")
            results['trend_changes'] = {'error': str(e)}
        
        # 3. Análisis de soporte y resistencia
        try:
            support_resistance = self.identify_support_resistance(
                price_series, window=min(20, len(price_series)//5)
            )
            
            # Convertir arrays numpy a listas para JSON
            if 'histogram_method' in support_resistance:
                for key in ['supports', 'resistances']:
                    if key in support_resistance['histogram_method']:
                        support_resistance['histogram_method'][key] = [
                            float(x) for x in support_resistance['histogram_method'][key]
                        ]
            
            if 'peaks_method' in support_resistance:
                for key in ['supports', 'resistances']:
                    if key in support_resistance['peaks_method']:
                        support_resistance['peaks_method'][key] = [
                            float(x) for x in support_resistance['peaks_method'][key]
                        ]
            
            results['support_resistance'] = support_resistance
        except Exception as e:
            logger.error(f"Error en análisis de soporte y resistencia: {str(e)}")
            results['support_resistance'] = {'error': str(e)}
        
        # 4. Análisis de estacionalidad
        try:
            seasonality = self.detect_seasonality(
                price_series, max_lag=min(365, len(price_series)//3)
            )
            results['seasonality'] = seasonality
        except Exception as e:
            logger.error(f"Error en análisis de estacionalidad: {str(e)}")
            results['seasonality'] = {'error': str(e)}
        
        # 5. Análisis de fase de mercado
        try:
            market_phase = self.analyze_market_phases(
                price_series, window=min(90, len(price_series)//3)
            )
            results['market_phase'] = market_phase
        except Exception as e:
            logger.error(f"Error en análisis de fase de mercado: {str(e)}")
            results['market_phase'] = {'error': str(e)}
        
        # 6. Verificación de estacionariedad
        try:
            stationarity = self.check_stationarity(
                price_series, window=min(30, len(price_series)//4)
            )
            results['stationarity'] = stationarity
        except Exception as e:
            logger.error(f"Error en verificación de estacionariedad: {str(e)}")
            results['stationarity'] = {'error': str(e)}
        
        # Guardar resultados si se especificó directorio
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{symbol.lower()}_trend_analysis.json")
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Resultados guardados en {output_file}")
            except Exception as e:
                logger.error(f"Error guardando resultados: {str(e)}")
        
        logger.info(f"Análisis de tendencias completado para {symbol}")
        return results
    
    def batch_analyze_trends(self, data_dir: str, assets: Dict[str, List[str]],
                           output_dir: str) -> Dict:
        """
        Ejecuta análisis de tendencias para múltiples activos.
        
        Args:
            data_dir: Directorio con datos procesados
            assets: Diccionario con listas de símbolos por tipo de activo
            output_dir: Directorio para guardar resultados
            
        Returns:
            Diccionario con resumen de resultados
        """
        os.makedirs(output_dir, exist_ok=True)
        results_summary = {}
        
        total_assets = sum(len(symbols) for symbols in assets.values())
        logger.info(f"Iniciando análisis por lotes para {total_assets} activos")
        
        for asset_type, symbols in assets.items():
            type_dir = os.path.join(output_dir, asset_type)
            os.makedirs(type_dir, exist_ok=True)
            
            type_results = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Analizando {symbol} ({asset_type})")
                    asset_results = self.run_trend_analysis(
                        data_dir, asset_type, symbol, type_dir
                    )
                    
                    if 'error' in asset_results:
                        type_results[symbol] = f"Error: {asset_results['error']}"
                    else:
                        # Crear un resumen simplificado
                        summary = {
                            'current_trend': asset_results.get('trend_direction', {}).get('current_trend'),
                            'trend_changes': len(asset_results.get('trend_changes', [])),
                            'market_phase': asset_results.get('market_phase', {}).get('current_phase'),
                            'confidence': asset_results.get('market_phase', {}).get('confidence'),
                            'has_seasonality': asset_results.get('seasonality', {}).get('has_significant_seasonality')
                        }
                        type_results[symbol] = summary
                        
                except Exception as e:
                    logger.error(f"Error procesando {symbol}: {str(e)}")
                    type_results[symbol] = f"Error: {str(e)}"
            
            results_summary[asset_type] = type_results
            
            # Guardar resumen por tipo de activo
            with open(os.path.join(type_dir, f"{asset_type}_summary.json"), 'w') as f:
                json.dump(type_results, f, indent=2)
        
        # Guardar resumen global
        with open(os.path.join(output_dir, "trend_analysis_summary.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Análisis por lotes completado para {total_assets} activos")
        return results_summary


if __name__ == "__main__":
    # Ejemplo de uso
    detector = TrendDetector()
    
    # Definir activos a analizar
    assets = {
        'stocks': ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        'crypto': ["BTC", "ETH", "SOL", "ADA", "DOGE"],
        'indices': ["^GSPC", "^DJI", "^IXIC"]
    }
    
    # Ejecutar análisis por lotes
    results = detector.batch_analyze_trends(
        data_dir="data",
        assets=assets,
        output_dir="data/trend_analysis"
    )
    
    print("Resumen de análisis:")
    for asset_type, symbols in results.items():
        print(f"\n{asset_type.upper()}:")
        for symbol, result in symbols.items():
            print(f"  {symbol}: {result}")