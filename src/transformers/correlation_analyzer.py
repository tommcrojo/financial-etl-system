#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para análisis de correlaciones entre diferentes tipos de activos financieros.
Permite analizar relaciones entre acciones, índices y criptomonedas.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/correlation_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('correlation_analyzer')


class CorrelationAnalyzer:
    """
    Clase para analizar correlaciones entre diferentes activos financieros.
    """
    
    def __init__(self):
        """Inicializa el analizador de correlaciones."""
        pass
    
    def load_data_for_correlation(self, data_dir: str, asset_type: str, 
                                symbols: List[str],
                                date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Carga datos para múltiples activos para análisis de correlación.
        
        Args:
            data_dir: Directorio con datos procesados
            asset_type: Tipo de activo ('stocks', 'crypto', 'indices')
            symbols: Lista de símbolos a incluir
            date_range: Rango de fechas (inicio, fin) como strings 'YYYY-MM-DD'
            
        Returns:
            DataFrame con precios/valores para todos los símbolos alineados por fecha
        """
        logger.info(f"Cargando datos para correlación: {asset_type}, {len(symbols)} símbolos")
        
        asset_data = {}
        valid_symbols = []
        
        # Determinar subcarpeta según tipo de activo
        if asset_type == 'stocks':
            subfolder = 'stocks/processed'
            file_suffix = '_processed.csv'
            value_column = 'close'
        elif asset_type == 'crypto':
            subfolder = 'crypto/processed'
            file_suffix = '_market_cycles.csv'  # Usamos el archivo más completo
            value_column = 'price'
        elif asset_type == 'indices':
            subfolder = 'stocks/processed'  # Los índices se tratan como acciones
            file_suffix = '_processed.csv'
            value_column = 'close'
        else:
            logger.error(f"Tipo de activo no válido: {asset_type}")
            return pd.DataFrame()
        
        full_data_dir = os.path.join(data_dir, subfolder)
        
        # Iterar sobre símbolos
        for symbol in symbols:
            symbol_lower = symbol.lower().replace('^', '')  # Manejar índices como ^GSPC
            file_path = os.path.join(full_data_dir, f"{symbol_lower}{file_suffix}")
            
            # Intentar archivo alternativo si no existe
            if not os.path.exists(file_path) and asset_type == 'crypto':
                file_path = os.path.join(full_data_dir, f"{symbol_lower}_indicators.csv")
                
                if not os.path.exists(file_path):
                    file_path = os.path.join(full_data_dir, f"{symbol_lower}_market_clean.csv")
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Verificar columnas necesarias
                    if 'date' not in df.columns or value_column not in df.columns:
                        logger.warning(f"Columnas necesarias no encontradas en {file_path}")
                        continue
                    
                    # Convertir fecha a datetime
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Filtrar por rango de fechas si se especifica
                    if date_range:
                        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    
                    # Verificar si hay suficientes datos
                    if len(df) < 5:
                        logger.warning(f"Insuficientes datos para {symbol}")
                        continue
                    
                    # Guardar solo fecha y valor
                    asset_data[symbol] = df[['date', value_column]].rename(
                        columns={value_column: symbol}
                    )
                    valid_symbols.append(symbol)
                    
                except Exception as e:
                    logger.error(f"Error cargando datos para {symbol}: {str(e)}")
            else:
                logger.warning(f"Archivo no encontrado: {file_path}")
        
        if not valid_symbols:
            logger.error("No se pudieron cargar datos válidos para ningún símbolo")
            return pd.DataFrame()
        
        logger.info(f"Datos cargados exitosamente para {len(valid_symbols)} símbolos")
        
        # Combinar datos de todos los símbolos
        combined_data = asset_data[valid_symbols[0]][['date']]
        
        for symbol in valid_symbols:
            combined_data = pd.merge(
                combined_data, 
                asset_data[symbol][['date', symbol]], 
                on='date', 
                how='outer'
            )
        
        # Ordenar por fecha
        combined_data = combined_data.sort_values('date').reset_index(drop=True)
        
        return combined_data
    
    def calculate_correlations(self, data: pd.DataFrame, method: str = 'pearson',
                             window: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Calcula matriz de correlación entre activos.
        
        Args:
            data: DataFrame con columnas de fecha y valores por activo
            method: Método de correlación ('pearson', 'spearman', 'kendall')
            window: Ventana para correlación móvil (None para correlación global)
            
        Returns:
            Tuple con (matriz de correlación, estadísticas adicionales)
        """
        # Excluir columna de fecha para cálculos
        value_data = data.drop('date', axis=1)
        
        # Eliminar filas con NaN
        value_data = value_data.dropna()
        
        if len(value_data) < 2:
            logger.error("Insuficientes datos para calcular correlaciones")
            return pd.DataFrame(), {}
        
        logger.info(f"Calculando correlaciones usando método {method}")
        
        stats_dict = {}
        
        # Calcular matriz de correlación global
        if window is None:
            corr_matrix = value_data.corr(method=method)
            
            # Calcular estadísticas adicionales
            stats_dict['avg_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            stats_dict['correlation_range'] = (
                corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min(),
                corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
            )
            
            # Identificar pares más y menos correlacionados
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlations.append({
                        'pair': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': corr_matrix.iloc[i, j]
                    })
            
            sorted_correlations = sorted(correlations, key=lambda x: x['correlation'])
            
            if sorted_correlations:
                stats_dict['least_correlated'] = sorted_correlations[0]
                stats_dict['most_correlated'] = sorted_correlations[-1]
            
            logger.info("Correlación global calculada exitosamente")
            return corr_matrix, stats_dict
            
        else:
            # Calcular correlación móvil
            rolling_corr = {}
            dates = data['date'].iloc[window-1:].reset_index(drop=True)
            
            # Preparar estructura para todas las correlaciones
            symbols = value_data.columns
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    pair_key = f"{symbols[i]}_{symbols[j]}"
                    rolling_corr[pair_key] = []
            
            # Calcular correlaciones móviles
            for i in range(len(value_data) - window + 1):
                window_data = value_data.iloc[i:i+window]
                window_corr = window_data.corr(method=method)
                
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        pair_key = f"{symbols[i]}_{symbols[j]}"
                        rolling_corr[pair_key].append(window_corr.loc[symbols[i], symbols[j]])
            
            # Convertir a DataFrame
            rolling_corr_df = pd.DataFrame(rolling_corr)
            rolling_corr_df['date'] = dates
            
            # Calcular estadísticas sobre correlaciones móviles
            stats_dict['avg_correlation_over_time'] = {
                pair: np.mean(values) for pair, values in rolling_corr.items()
            }
            
            stats_dict['correlation_volatility'] = {
                pair: np.std(values) for pair, values in rolling_corr.items()
            }
            
            # Identificar pares con mayor cambio en correlación
            correlation_changes = {
                pair: np.max(values) - np.min(values) for pair, values in rolling_corr.items()
            }
            
            stats_dict['max_correlation_change'] = max(
                correlation_changes.items(), key=lambda x: x[1]
            )
            
            logger.info(f"Correlación móvil (ventana={window}) calculada exitosamente")
            
            # Devolver última matriz de correlación y estadísticas
            last_corr_matrix = value_data.iloc[-window:].corr(method=method)
            
            return last_corr_matrix, stats_dict
    
    def analyze_correlations_by_period(self, data: pd.DataFrame, 
                                     periods: List[Tuple[str, str]],
                                     method: str = 'pearson') -> Dict[str, pd.DataFrame]:
        """
        Analiza correlaciones divididas por períodos específicos.
        
        Args:
            data: DataFrame con columnas de fecha y valores por activo
            periods: Lista de tuplas (nombre_periodo, fecha_inicio, fecha_fin)
            method: Método de correlación ('pearson', 'spearman', 'kendall')
            
        Returns:
            Diccionario con matriz de correlación por período
        """
        result = {}
        
        logger.info(f"Analizando correlaciones para {len(periods)} períodos")
        
        for period_name, start_date, end_date in periods:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filtrar datos para el período
            period_data = data[
                (data['date'] >= start_dt) & 
                (data['date'] <= end_dt)
            ].copy()
            
            if len(period_data) < 5:
                logger.warning(f"Insuficientes datos para período {period_name}")
                continue
            
            # Calcular correlación para el período
            period_data_values = period_data.drop('date', axis=1)
            corr_matrix = period_data_values.corr(method=method)
            
            result[period_name] = corr_matrix
            logger.info(f"Calculada correlación para período {period_name} ({len(period_data)} registros)")
        
        return result
    
    def detect_correlation_regime_changes(self, data: pd.DataFrame, window: int = 30,
                                        threshold: float = 0.3) -> List[Dict]:
        """
        Detecta cambios significativos en regímenes de correlación.
        
        Args:
            data: DataFrame con columnas de fecha y valores por activo
            window: Tamaño de ventana móvil para comparar correlaciones
            threshold: Umbral para considerar cambio significativo
            
        Returns:
            Lista de diccionarios con fechas y detalles de cambios detectados
        """
        logger.info(f"Buscando cambios de régimen de correlación (umbral={threshold})")
        
        # Necesitamos al menos 2*window registros
        if len(data) < 2*window:
            logger.error(f"Insuficientes datos para detectar cambios de régimen. Requeridos: {2*window}, disponibles: {len(data)}")
            return []
        
        changes = []
        dates = data['date'].iloc[window:].reset_index(drop=True)
        value_data = data.drop('date', axis=1)
        
        # Calcular matrices de correlación móviles
        for i in range(len(value_data) - 2*window + 1):
            window1 = value_data.iloc[i:i+window]
            window2 = value_data.iloc[i+window:i+2*window]
            
            corr1 = window1.corr()
            corr2 = window2.corr()
            
            # Extraer valores de triangular superior para comparar
            triu_indices = np.triu_indices_from(corr1.values, k=1)
            corr1_values = corr1.values[triu_indices]
            corr2_values = corr2.values[triu_indices]
            
            # Calcular diferencia absoluta promedio entre matrices
            diff = np.mean(np.abs(corr1_values - corr2_values))
            
            if diff > threshold:
                # Identificar los pares que más cambiaron
                pair_changes = []
                for a in range(len(corr1.columns)):
                    for b in range(a+1, len(corr1.columns)):
                        col1, col2 = corr1.columns[a], corr1.columns[b]
                        change = abs(corr1.loc[col1, col2] - corr2.loc[col1, col2])
                        
                        if change > threshold:
                            pair_changes.append({
                                'pair': (col1, col2),
                                'before': corr1.loc[col1, col2],
                                'after': corr2.loc[col1, col2],
                                'change': change
                            })
                
                # Ordenar cambios por magnitud
                pair_changes = sorted(pair_changes, key=lambda x: x['change'], reverse=True)
                
                # Registrar el cambio
                changes.append({
                    'date': dates.iloc[i],
                    'correlation_difference': diff,
                    'significant_pair_changes': pair_changes[:5]  # Top 5 cambios
                })
        
        # Filtrar para evitar cambios demasiado cercanos (dentro de window/2 días)
        if changes:
            filtered_changes = [changes[0]]
            
            for change in changes[1:]:
                days_since_last = (change['date'] - filtered_changes[-1]['date']).days
                
                if days_since_last > window/2:
                    filtered_changes.append(change)
            
            logger.info(f"Detectados {len(filtered_changes)} cambios de régimen de correlación significativos")
            return filtered_changes
        else:
            logger.info("No se detectaron cambios de régimen de correlación significativos")
            return []
    
    def perform_cluster_analysis(self, corr_matrix: pd.DataFrame, n_clusters: int = 3) -> Dict:
        """
        Realiza análisis de clusters para identificar grupos de activos correlacionados.
        
        Args:
            corr_matrix: Matriz de correlación
            n_clusters: Número de clusters a identificar
            
        Returns:
            Diccionario con resultados del análisis de clusters
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            logger.info(f"Realizando análisis de clusters con n_clusters={n_clusters}")
            
            # Convertir matriz de correlación a matriz de distancia
            # 1 - abs(corr) para que valores altamente correlacionados (+ o -) estén cerca
            distance_matrix = 1 - np.abs(corr_matrix.values)
            
            # Aplicar clustering jerárquico
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            )
            
            labels = clustering.fit_predict(distance_matrix)
            
            # Organizar resultados
            clusters = {}
            for i, label in enumerate(labels):
                symbol = corr_matrix.columns[i]
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(symbol)
            
            # Calcular correlación media intra-cluster
            intra_cluster_corr = {}
            for cluster_id, symbols in clusters.items():
                if len(symbols) > 1:
                    # Extraer submatriz para este cluster
                    cluster_matrix = corr_matrix.loc[symbols, symbols]
                    
                    # Calcular correlación media (solo triangular superior)
                    triu_indices = np.triu_indices_from(cluster_matrix.values, k=1)
                    if len(triu_indices[0]) > 0:  # Si hay más de un símbolo
                        mean_corr = np.mean(cluster_matrix.values[triu_indices])
                        intra_cluster_corr[cluster_id] = mean_corr
                    else:
                        intra_cluster_corr[cluster_id] = 1.0  # Solo un símbolo
                else:
                    intra_cluster_corr[cluster_id] = 1.0  # Solo un símbolo
            
            results = {
                'clusters': clusters,
                'intra_cluster_correlation': intra_cluster_corr,
                'labels': labels.tolist()
            }
            
            logger.info(f"Análisis de clusters completado: {n_clusters} clusters identificados")
            return results
            
        except ImportError:
            logger.error("sklearn.cluster no está disponible para análisis de clusters")
            return {
                'error': 'sklearn.cluster no disponible',
                'clusters': {},
                'intra_cluster_correlation': {}
            }
    
    def perform_pca_analysis(self, data: pd.DataFrame, n_components: int = 2) -> Dict:
        """
        Realiza análisis de componentes principales (PCA) para identificar factores comunes.
        
        Args:
            data: DataFrame con valores por activo (sin columna de fecha)
            n_components: Número de componentes principales a extraer
            
        Returns:
            Diccionario con resultados del análisis PCA
        """
        if 'date' in data.columns:
            data = data.drop('date', axis=1)
        
        # Eliminar filas con valores faltantes
        data = data.dropna()
        
        if len(data) < 10:
            logger.error("Insuficientes datos para análisis PCA")
            return {'error': 'Insuficientes datos'}
        
        logger.info(f"Realizando análisis PCA con {n_components} componentes")
        
        # Estandarizar datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data_scaled)
        
        # Organizar resultados
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_explained_variance': sum(pca.explained_variance_ratio_),
            'components': pca.components_.tolist(),
            'feature_names': data.columns.tolist()
        }
        
        # Crear DataFrame con componentes principales
        pc_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Añadir contribución de cada activo a los componentes
        contributions = {}
        for i, component in enumerate(pca.components_):
            pc_name = f'PC{i+1}'
            contributions[pc_name] = {
                feature: abs(weight) for feature, weight in zip(data.columns, component)
            }
            
            # Ordenar por contribución
            contributions[pc_name] = dict(
                sorted(contributions[pc_name].items(), key=lambda x: x[1], reverse=True)
            )
        
        pca_results['contributions'] = contributions
        
        logger.info(f"Análisis PCA completado: {pca_results['total_explained_variance']*100:.2f}% de varianza explicada")
        return pca_results
    
    def analyze_cross_asset_correlations(self, data_dir: str, 
                                       stocks: List[str], 
                                       crypto: List[str],
                                       indices: List[str],
                                       date_range: Optional[Tuple[str, str]] = None) -> Dict:
        """
        Analiza correlaciones entre diferentes clases de activos.
        
        Args:
            data_dir: Directorio con datos procesados
            stocks: Lista de símbolos de acciones
            crypto: Lista de símbolos de criptomonedas
            indices: Lista de símbolos de índices
            date_range: Rango de fechas como tupla (inicio, fin)
            
        Returns:
            Diccionario con análisis de correlaciones entre clases de activos
        """
        logger.info("Iniciando análisis de correlaciones entre clases de activos")
        
        results = {
            'stocks_vs_crypto': {},
            'stocks_vs_indices': {},
            'crypto_vs_indices': {},
            'intra_stock_corr': {},
            'intra_crypto_corr': {},
            'intra_indices_corr': {}
        }
        
        # 1. Cargar datos para cada tipo de activo
        stocks_data = self.load_data_for_correlation(data_dir, 'stocks', stocks, date_range)
        crypto_data = self.load_data_for_correlation(data_dir, 'crypto', crypto, date_range)
        indices_data = self.load_data_for_correlation(data_dir, 'indices', indices, date_range)
        
        # Verificar que tenemos datos suficientes
        if stocks_data.empty or crypto_data.empty or indices_data.empty:
            logger.error("Datos insuficientes para al menos un tipo de activo")
            return {'error': 'Datos insuficientes'}
        
        # 2. Calcular correlaciones internas para cada clase
        if len(stocks_data.columns) > 2:  # date + al menos 2 símbolos
            corr_matrix, stats = self.calculate_correlations(stocks_data)
            results['intra_stock_corr'] = {
                'matrix': corr_matrix.to_dict(),
                'stats': stats
            }
        
        if len(crypto_data.columns) > 2:
            corr_matrix, stats = self.calculate_correlations(crypto_data)
            results['intra_crypto_corr'] = {
                'matrix': corr_matrix.to_dict(),
                'stats': stats
            }
        
        if len(indices_data.columns) > 2:
            corr_matrix, stats = self.calculate_correlations(indices_data)
            results['intra_indices_corr'] = {
                'matrix': corr_matrix.to_dict(),
                'stats': stats
            }
        
        # 3. Combinar datos entre clases de activos para análisis cruzado
        
        # Stocks vs Crypto
        if not stocks_data.empty and not crypto_data.empty:
            combined = pd.merge(
                stocks_data, crypto_data.drop(columns=stocks_data.columns.intersection(crypto_data.columns).difference(['date'])),
                on='date', how='inner'
            )
            
            if len(combined) > 10:
                corr_matrix, stats = self.calculate_correlations(combined)
                
                # Extraer solo correlaciones entre clases
                stock_cols = [col for col in stocks_data.columns if col != 'date']
                crypto_cols = [col for col in crypto_data.columns if col != 'date' and col not in stock_cols]
                
                cross_corr = corr_matrix.loc[stock_cols, crypto_cols]
                
                results['stocks_vs_crypto'] = {
                    'matrix': cross_corr.to_dict(),
                    'avg_correlation': cross_corr.values.mean(),
                    'max_correlation': (
                        cross_corr.max().max(),
                        cross_corr.max().idxmax(),
                        cross_corr.max(axis=1).idxmax()
                    ),
                    'min_correlation': (
                        cross_corr.min().min(),
                        cross_corr.min().idxmin(),
                        cross_corr.min(axis=1).idxmin()
                    )
                }
        
        # Análisis similar para Stocks vs Indices y Crypto vs Indices
        # Stocks vs Indices
        if not stocks_data.empty and not indices_data.empty:
            combined = pd.merge(
                stocks_data, indices_data.drop(columns=stocks_data.columns.intersection(indices_data.columns).difference(['date'])),
                on='date', how='inner'
            )
            
            if len(combined) > 10:
                corr_matrix, stats = self.calculate_correlations(combined)
                
                stock_cols = [col for col in stocks_data.columns if col != 'date']
                indices_cols = [col for col in indices_data.columns if col != 'date' and col not in stock_cols]
                
                cross_corr = corr_matrix.loc[stock_cols, indices_cols]
                
                results['stocks_vs_indices'] = {
                    'matrix': cross_corr.to_dict(),
                    'avg_correlation': cross_corr.values.mean(),
                    'max_correlation': (
                        cross_corr.max().max(),
                        cross_corr.max().idxmax(),
                        cross_corr.max(axis=1).idxmax()
                    ),
                    'min_correlation': (
                        cross_corr.min().min(),
                        cross_corr.min().idxmin(),
                        cross_corr.min(axis=1).idxmin()
                    )
                }
        
        # Crypto vs Indices
        if not crypto_data.empty and not indices_data.empty:
            combined = pd.merge(
                crypto_data, indices_data.drop(columns=crypto_data.columns.intersection(indices_data.columns).difference(['date'])),
                on='date', how='inner'
            )
            
            if len(combined) > 10:
                corr_matrix, stats = self.calculate_correlations(combined)
                
                crypto_cols = [col for col in crypto_data.columns if col != 'date']
                indices_cols = [col for col in indices_data.columns if col != 'date' and col not in crypto_cols]
                
                cross_corr = corr_matrix.loc[crypto_cols, indices_cols]
                
                results['crypto_vs_indices'] = {
                    'matrix': cross_corr.to_dict(),
                    'avg_correlation': cross_corr.values.mean(),
                    'max_correlation': (
                        cross_corr.max().max(),
                        cross_corr.max().idxmax(),
                        cross_corr.max(axis=1).idxmax()
                    ),
                    'min_correlation': (
                        cross_corr.min().min(),
                        cross_corr.min().idxmin(),
                        cross_corr.min(axis=1).idxmin()
                    )
                }
        
        # 4. Análisis de cambios en correlaciones durante períodos específicos
        if not stocks_data.empty and not crypto_data.empty and date_range:
            try:
                # Intentar inferir períodos importantes (e.g., pre/post COVID)
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                mid_point = start_date + (end_date - start_date) / 2
                
                periods = [
                    ('first_half', start_date.strftime('%Y-%m-%d'), mid_point.strftime('%Y-%m-%d')),
                    ('second_half', mid_point.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                ]
                
                # Combinar todos los datos
                all_data = pd.merge(stocks_data, crypto_data.drop(columns=stocks_data.columns.intersection(crypto_data.columns).difference(['date'])),
                                   on='date', how='outer')
                
                if not indices_data.empty:
                    all_data = pd.merge(all_data, indices_data.drop(columns=all_data.columns.intersection(indices_data.columns).difference(['date'])),
                                       on='date', how='outer')
                
                # Analizar correlaciones por período
                period_correlations = self.analyze_correlations_by_period(all_data, periods)
                
                # Calcular cambios en correlaciones
                if len(period_correlations) > 1:
                    correlation_changes = {}
                    for asset1 in all_data.columns:
                        if asset1 == 'date':
                            continue
                            
                        for asset2 in all_data.columns:
                            if asset2 == 'date' or asset2 <= asset1:  # Evitar duplicados
                                continue
                            
                            pair = f"{asset1}_{asset2}"
                            correlation_changes[pair] = {}
                            
                            for period in period_correlations:
                                if asset1 in period_correlations[period].columns and asset2 in period_correlations[period].columns:
                                    correlation_changes[pair][period] = period_correlations[period].loc[asset1, asset2]
                    
                    results['correlation_changes_by_period'] = correlation_changes
            
            except Exception as e:
                logger.error(f"Error en análisis por períodos: {str(e)}")
        
        logger.info("Análisis de correlaciones entre clases de activos completado")
        return results
    
    def run_comprehensive_correlation_analysis(self, data_dir: str, 
                                             stocks: List[str], 
                                             crypto: List[str],
                                             indices: List[str],
                                             output_dir: str,
                                             date_range: Optional[Tuple[str, str]] = None) -> Dict:
        """
        Ejecuta un análisis completo de correlaciones y guarda resultados.
        
        Args:
            data_dir: Directorio con datos procesados
            stocks: Lista de símbolos de acciones
            crypto: Lista de símbolos de criptomonedas
            indices: Lista de símbolos de índices
            output_dir: Directorio para guardar resultados
            date_range: Rango de fechas como tupla (inicio, fin)
            
        Returns:
            Diccionario con resumen de resultados
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Iniciando análisis completo de correlaciones")
        results_summary = {}
        
        # 1. Análisis de correlaciones entre clases de activos
        cross_asset_results = self.analyze_cross_asset_correlations(
            data_dir, stocks, crypto, indices, date_range
        )
        
        if 'error' not in cross_asset_results:
            # Guardar resultados
            with open(os.path.join(output_dir, "cross_asset_correlations.json"), 'w') as f:
                json.dump(cross_asset_results, f, indent=2)
            
            results_summary['cross_asset_analysis'] = "Completado y guardado"
        else:
            results_summary['cross_asset_analysis'] = f"Error: {cross_asset_results.get('error')}"
        
        # 2. Cargar todos los datos para análisis global
        all_symbols = stocks + crypto + indices
        all_data = pd.DataFrame(columns=['date'])
        
        for asset_type, symbols in [('stocks', stocks), ('crypto', crypto), ('indices', indices)]:
            asset_data = self.load_data_for_correlation(data_dir, asset_type, symbols, date_range)
            if not asset_data.empty:
                # Fusionar con datos previos
                all_data = pd.merge(
                    all_data,
                    asset_data.drop(columns=all_data.columns.intersection(asset_data.columns).difference(['date'])),
                    on='date', how='outer'
                )
        
        if len(all_data.columns) <= 2:  # Solo fecha y quizás un activo
            logger.error("Insuficientes datos para análisis global")
            results_summary['global_analysis'] = "Error: Insuficientes datos"
            return results_summary
        
        # 3. Análisis de correlación global
        try:
            # Matriz de correlación global
            corr_matrix, stats = self.calculate_correlations(all_data)
            
            # Guardar matriz de correlación
            corr_matrix.to_csv(os.path.join(output_dir, "global_correlation_matrix.csv"))
            
            with open(os.path.join(output_dir, "correlation_stats.json"), 'w') as f:
                # Convertir valores NumPy a Python nativos para JSON
                stats_json = {}
                for key, value in stats.items():
                    if isinstance(value, dict):
                        stats_json[key] = {k: v for k, v in value.items()}
                    elif isinstance(value, (list, tuple)) and all(isinstance(x, np.ndarray) for x in value):
                        stats_json[key] = [x.item() if hasattr(x, 'item') else x for x in value]
                    elif hasattr(value, 'item'):  # Para valores NumPy
                        stats_json[key] = value.item()
                    else:
                        stats_json[key] = value
                
                json.dump(stats_json, f, indent=2)
            
            results_summary['global_correlation'] = "Completado y guardado"
            
            # 4. Análisis de clusters
            if len(all_data.columns) > 3:  # fecha + al menos 3 activos
                n_clusters = min(5, len(all_data.columns) - 1)  # No más de 5 clusters
                cluster_results = self.perform_cluster_analysis(corr_matrix, n_clusters=n_clusters)
                
                with open(os.path.join(output_dir, "correlation_clusters.json"), 'w') as f:
                    json.dump(cluster_results, f, indent=2)
                
                results_summary['cluster_analysis'] = "Completado y guardado"
            
            # 5. Análisis de componentes principales
            if len(all_data.columns) > 3:  # fecha + al menos 3 activos
                n_components = min(3, len(all_data.columns) - 1)
                pca_results = self.perform_pca_analysis(all_data, n_components=n_components)
                
                with open(os.path.join(output_dir, "pca_analysis.json"), 'w') as f:
                    json.dump(pca_results, f, indent=2)
                
                results_summary['pca_analysis'] = "Completado y guardado"
            
            # 6. Detección de cambios de régimen
            if len(all_data) > 60:  # Al menos 60 días de datos
                regime_changes = self.detect_correlation_regime_changes(
                    all_data, window=30, threshold=0.3
                )
                
                if regime_changes:
                    # Convertir fechas a string para JSON
                    for change in regime_changes:
                        change['date'] = change['date'].strftime('%Y-%m-%d')
                    
                    with open(os.path.join(output_dir, "correlation_regime_changes.json"), 'w') as f:
                        json.dump(regime_changes, f, indent=2)
                    
                    results_summary['regime_change_detection'] = f"Completado: {len(regime_changes)} cambios detectados"
                else:
                    results_summary['regime_change_detection'] = "Completado: No se detectaron cambios significativos"
            
            # 7. Correlaciones móviles para principales pares
            if len(all_data) > 90:  # Al menos 90 días
                # Seleccionar los pares más interesantes (mayor volatilidad en correlación)
                rolling_window = 30
                _, rolling_stats = self.calculate_correlations(all_data, window=rolling_window)
                
                if 'correlation_volatility' in rolling_stats:
                    # Seleccionar los 5 pares con mayor volatilidad
                    top_volatile_pairs = sorted(
                        rolling_stats['correlation_volatility'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    # Calcular correlaciones móviles por par
                    rolling_correlations = {}
                    
                    for pair, _ in top_volatile_pairs:
                        asset1, asset2 = pair.split('_')
                        pair_data = all_data[['date', asset1, asset2]].dropna()
                        
                        if len(pair_data) > rolling_window:
                            # Calcular correlación móvil
                            rolling_corr = []
                            for i in range(len(pair_data) - rolling_window + 1):
                                window_data = pair_data.iloc[i:i+rolling_window, 1:]  # Excluir fecha
                                corr = window_data.corr().iloc[0, 1]  # Correlación entre los dos activos
                                rolling_corr.append({
                                    'date': pair_data.iloc[i+rolling_window-1]['date'].strftime('%Y-%m-%d'),
                                    'correlation': corr
                                })
                            
                            rolling_correlations[pair] = rolling_corr
                    
                    with open(os.path.join(output_dir, "rolling_correlations.json"), 'w') as f:
                        json.dump(rolling_correlations, f, indent=2)
                    
                    results_summary['rolling_correlations'] = "Completado para los pares más volátiles"
        
        except Exception as e:
            logger.error(f"Error en análisis global: {str(e)}")
            results_summary['global_analysis'] = f"Error: {str(e)}"
        
        # Guardar resumen de resultados
        with open(os.path.join(output_dir, "correlation_analysis_summary.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info("Análisis completo de correlaciones finalizado")
        return results_summary


if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = CorrelationAnalyzer()
    
    # Definir activos a analizar
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    crypto = ["BTC", "ETH", "SOL", "ADA", "DOGE"]
    indices = ["^GSPC", "^DJI", "^IXIC"]
    
    # Ejecutar análisis completo
    results = analyzer.run_comprehensive_correlation_analysis(
        data_dir="data",
        stocks=stocks,
        crypto=crypto,
        indices=indices,
        output_dir="data/correlation_analysis",
        date_range=("2022-01-01", "2024-07-31")
    )
    
    print("Resumen de análisis:")
    for key, value in results.items():
        print(f"{key}: {value}")