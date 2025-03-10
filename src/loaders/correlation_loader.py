#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para cargar datos de correlaciones en la base de datos.
Procesa datos de correlación entre diferentes activos.
"""

import os
import sys
import json
import logging
import glob
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, and_, or_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from dotenv import load_dotenv

# Agregar directorio raíz al path para importar modelos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.create_database_schema import Asset, CorrelationData, ETLLog, DB_URL

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/correlation_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('correlation_loader')

# Cargar variables de entorno
load_dotenv()


class CorrelationLoader:
    """
    Clase para cargar datos de correlaciones en la base de datos.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Inicializa el cargador de datos.
        
        Args:
            db_url: URL de conexión a la base de datos (opcional)
        """
        self.db_url = db_url or DB_URL
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_asset_id_map(self) -> Dict[str, int]:
        """
        Obtiene un mapeo de símbolos a IDs de activos.
        
        Returns:
            Diccionario con mapeo símbolo -> ID
        """
        session = self.Session()
        try:
            # Obtener todos los activos
            assets = session.query(Asset.asset_id, Asset.symbol).all()
            
            # Crear mapeo
            asset_map = {asset.symbol: asset.asset_id for asset in assets}
            
            logger.info(f"Obtenidos {len(asset_map)} activos para mapeo")
            return asset_map
            
        except Exception as e:
            logger.error(f"Error obteniendo mapeo de activos: {str(e)}")
            return {}
        finally:
            session.close()
    
    def load_correlation_matrix(self, correlation_matrix: pd.DataFrame, 
                              calc_date: Union[str, datetime],
                              time_window: str = '30d') -> Tuple[int, int, int]:
        """
        Carga una matriz de correlación en la base de datos.
        
        Args:
            correlation_matrix: DataFrame con matriz de correlación
            calc_date: Fecha de cálculo
            time_window: Ventana de tiempo para la correlación ('30d', '90d', etc.)
            
        Returns:
            Tupla (registros_insertados, registros_actualizados, registros_ignorados)
        """
        if correlation_matrix.empty:
            logger.warning("Matriz de correlación vacía, no hay datos para cargar")
            return (0, 0, 0)
        
        # Convertir fecha a datetime si es string
        if isinstance(calc_date, str):
            calc_date = pd.to_datetime(calc_date)
        
        # Obtener mapeo de símbolos a IDs
        asset_map = self.get_asset_id_map()
        
        if not asset_map:
            logger.error("No se pudo obtener mapeo de activos")
            return (0, 0, 0)
        
        # Preparar registros para inserción
        records = []
        now = datetime.now()
        
        # Recorrer la matriz triangular superior
        for i, asset1 in enumerate(correlation_matrix.columns):
            for j, asset2 in enumerate(correlation_matrix.columns):
                # Solo procesar triangular superior (evitar duplicados)
                if j <= i:
                    continue
                
                # Verificar que ambos activos existen en el mapeo
                if asset1 not in asset_map or asset2 not in asset_map:
                    logger.warning(f"Activos no encontrados en BD: {asset1} o {asset2}")
                    continue
                
                # Obtener valor de correlación
                correlation_value = correlation_matrix.loc[asset1, asset2]
                
                # Ignorar valores nulos
                if pd.isna(correlation_value):
                    continue
                
                # Crear registro
                records.append({
                    'asset_id_1': asset_map[asset1],
                    'asset_id_2': asset_map[asset2],
                    'calc_date': calc_date,
                    'time_window': time_window,
                    'correlation_value': float(correlation_value),
                    'created_at': now
                })
        
        # Cargar en la base de datos
        session = self.Session()
        inserted, updated, ignored = 0, 0, 0
        
        try:
            # Procesar todos los registros
            for record in records:
                try:
                    # Verificar si ya existe
                    existing = session.query(CorrelationData).filter(
                        and_(
                            CorrelationData.asset_id_1 == record['asset_id_1'],
                            CorrelationData.asset_id_2 == record['asset_id_2'],
                            CorrelationData.calc_date == record['calc_date'],
                            CorrelationData.time_window == record['time_window']
                        )
                    ).first()
                    
                    if existing:
                        # Actualizar si el valor es diferente
                        if existing.correlation_value != record['correlation_value']:
                            existing.correlation_value = record['correlation_value']
                            updated += 1
                        else:
                            ignored += 1
                    else:
                        # Insertar nuevo registro
                        session.add(CorrelationData(**record))
                        inserted += 1
                    
                except IntegrityError:
                    session.rollback()
                    ignored += 1
                    continue
                
                # Commit cada 100 registros para reducir la presión sobre la BD
                if (inserted + updated + ignored) % 100 == 0:
                    session.commit()
            
            # Commit final
            session.commit()
            
            logger.info(f"Carga de correlaciones completada: {inserted} insertadas, {updated} actualizadas, {ignored} ignoradas")
            return (inserted, updated, ignored)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cargando correlaciones: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_correlation_data(self, file_path: str, date_column: Optional[str] = None) -> Dict:
        """
        Carga datos de correlación desde un archivo.
        
        Args:
            file_path: Ruta al archivo (CSV o JSON)
            date_column: Nombre de la columna de fecha (para archivos CSV con múltiples fechas)
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                return self._load_from_csv(file_path, date_column)
            elif file_extension == '.json':
                return self._load_from_json(file_path)
            else:
                logger.error(f"Formato de archivo no soportado: {file_extension}")
                return {
                    'status': 'error',
                    'message': f'Formato de archivo no soportado: {file_extension}'
                }
                
        except Exception as e:
            logger.error(f"Error cargando datos de correlación desde {file_path}: {str(e)}")
            
            # Registrar error en log
            self._log_process(
                process_name=f"load_correlation_{os.path.basename(file_path)}",
                status="failed",
                error_message=str(e),
                records_processed=0,
                execution_duration=None
            )
            
            return {
                'file': file_path,
                'status': 'error',
                'message': str(e)
            }
    
    def _load_from_csv(self, file_path: str, date_column: Optional[str] = None) -> Dict:
        """
        Carga datos de correlación desde un archivo CSV.
        
        Args:
            file_path: Ruta al archivo CSV
            date_column: Nombre de la columna de fecha
            
        Returns:
            Diccionario con resultados
        """
        logger.info(f"Cargando correlaciones desde CSV: {file_path}")
        
        # Cargar CSV
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.warning(f"Archivo CSV vacío: {file_path}")
            return {
                'status': 'warning',
                'message': 'Archivo CSV vacío',
                'inserted': 0,
                'updated': 0,
                'ignored': 0
            }
        
        # Si el CSV es una matriz de correlación (la primera columna son los activos)
        if date_column is None:
            # Verificar si la primera columna podría ser de activos
            first_col = df.columns[0]
            
            # Si la primera columna tiene valor numérico en el encabezado, 
            # probablemente sea una matriz de correlación sin nombres de fila
            if not isinstance(first_col, str) or first_col.replace('.', '', 1).isdigit():
                logger.warning(f"CSV no parece tener nombres de fila, generando nombres")
                # Generar nombres de fila: Asset1, Asset2, etc.
                df.insert(0, 'Asset', [f'Asset{i+1}' for i in range(len(df))])
                first_col = 'Asset'
            
            # Crear matriz de correlación con la primera columna como índice
            df_corr = df.set_index(first_col)
            
            # Si las columnas son strings, usarlas como nombres de activos
            # En caso contrario, usar los mismos nombres que en el índice
            if all(isinstance(col, str) for col in df_corr.columns):
                # Asegurarse de que los nombres de columnas no son numéricos
                if not all(col.replace('.', '', 1).isdigit() for col in df_corr.columns):
                    # OK, usar columnas como están
                    pass
                else:
                    # Columnas son numéricas, usar los mismos nombres que en el índice
                    df_corr.columns = df_corr.index
            else:
                # Columnas no son strings, usar los mismos nombres que en el índice
                df_corr.columns = df_corr.index
            
            # Usar fecha actual como fecha de cálculo
            calc_date = datetime.now()
            
            # Cargar la matriz
            inserted, updated, ignored = self.load_correlation_matrix(
                df_corr, calc_date, time_window='custom'
            )
            
            # Registrar en log
            self._log_process(
                process_name=f"load_correlation_matrix_{os.path.basename(file_path)}",
                status="success",
                records_processed=inserted + updated,
                execution_duration=None
            )
            
            return {
                'status': 'success',
                'file': file_path,
                'date': calc_date.strftime('%Y-%m-%d'),
                'inserted': inserted,
                'updated': updated,
                'ignored': ignored
            }
            
        else:
            # CSV con múltiples fechas
            if date_column not in df.columns:
                logger.error(f"Columna de fecha '{date_column}' no encontrada en CSV")
                return {
                    'status': 'error',
                    'message': f"Columna de fecha '{date_column}' no encontrada"
                }
            
            # Convertir columna de fecha a datetime
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Agrupar por fecha
            grouped = df.groupby(date_column)
            
            total_inserted, total_updated, total_ignored = 0, 0, 0
            
            for date, group in grouped:
                # Para cada fecha, crear matriz de correlación
                # (necesitaríamos conocer la estructura específica del CSV)
                # ...
                
                logger.warning(f"Carga de CSV con múltiples fechas no implementada completamente")
                return {
                    'status': 'error',
                    'message': 'Carga de CSV con múltiples fechas no implementada completamente'
                }
            
            return {
                'status': 'success',
                'file': file_path,
                'dates_processed': len(grouped),
                'inserted': total_inserted,
                'updated': total_updated,
                'ignored': total_ignored
            }
    
    def _load_from_json(self, file_path: str) -> Dict:
        """
        Carga datos de correlación desde un archivo JSON.
        
        Args:
            file_path: Ruta al archivo JSON
            
        Returns:
            Diccionario con resultados
        """
        logger.info(f"Cargando correlaciones desde JSON: {file_path}")
        
        # Cargar JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            logger.warning(f"Archivo JSON vacío: {file_path}")
            return {
                'status': 'warning',
                'message': 'Archivo JSON vacío',
                'inserted': 0,
                'updated': 0,
                'ignored': 0
            }
        
        # Procesar según estructura del JSON
        total_inserted, total_updated, total_ignored = 0, 0, 0
        processed_matrices = 0
        
        # Buscar matrices de correlación en varias posibles estructuras
        # 1. Estructura simple: {'matrix': {...}}
        if 'matrix' in data and isinstance(data['matrix'], dict):
            # Convertir a DataFrame
            matrix_df = pd.DataFrame.from_dict(data['matrix'])
            
            # Fecha y ventana
            calc_date = data.get('calc_date', datetime.now())
            if isinstance(calc_date, str):
                calc_date = pd.to_datetime(calc_date)
            time_window = data.get('time_window', 'custom')
            
            # Cargar
            inserted, updated, ignored = self.load_correlation_matrix(
                matrix_df, calc_date, time_window
            )
            
            total_inserted += inserted
            total_updated += updated
            total_ignored += ignored
            processed_matrices += 1
            
        # 2. Estructura por tipo de activo: {'stocks_vs_crypto': {'matrix': {...}}}
        correlation_types = [
            'stocks_vs_crypto', 'stocks_vs_indices', 'crypto_vs_indices',
            'intra_stock_corr', 'intra_crypto_corr', 'intra_indices_corr',
            'global_correlation'
        ]
        
        for corr_type in correlation_types:
            if corr_type in data and 'matrix' in data[corr_type]:
                matrix_data = data[corr_type]['matrix']
                
                # Convertir a DataFrame
                matrix_df = pd.DataFrame.from_dict(matrix_data)
                
                # Fecha y ventana
                calc_date = data.get('calc_date', datetime.now())
                if isinstance(calc_date, str):
                    calc_date = pd.to_datetime(calc_date)
                time_window = f"{corr_type}_custom"
                
                # Cargar
                inserted, updated, ignored = self.load_correlation_matrix(
                    matrix_df, calc_date, time_window
                )
                
                total_inserted += inserted
                total_updated += updated
                total_ignored += ignored
                processed_matrices += 1
        
        # 3. Estructura de rolling_correlations
        if 'rolling_correlations' in data and isinstance(data['rolling_correlations'], dict):
            # Para cada par de activos
            for pair, correlations in data['rolling_correlations'].items():
                assets = pair.split('_')
                if len(assets) != 2:
                    logger.warning(f"Formato de par inválido: {pair}")
                    continue
                
                asset1, asset2 = assets
                
                # Obtener mapeo de activos
                asset_map = self.get_asset_id_map()
                
                if asset1 not in asset_map or asset2 not in asset_map:
                    logger.warning(f"Activos no encontrados en BD: {asset1} o {asset2}")
                    continue
                
                # Cargar cada correlación
                for corr_data in correlations:
                    if 'date' in corr_data and 'correlation' in corr_data:
                        try:
                            date = pd.to_datetime(corr_data['date'])
                            value = float(corr_data['correlation'])
                            
                            # Crear registro
                            record = {
                                'asset_id_1': asset_map[asset1],
                                'asset_id_2': asset_map[asset2],
                                'calc_date': date,
                                'time_window': 'rolling_30d',  # Valor por defecto
                                'correlation_value': value,
                                'created_at': datetime.now()
                            }
                            
                            # Insertar o actualizar
                            session = self.Session()
                            try:
                                existing = session.query(CorrelationData).filter(
                                    and_(
                                        CorrelationData.asset_id_1 == record['asset_id_1'],
                                        CorrelationData.asset_id_2 == record['asset_id_2'],
                                        CorrelationData.calc_date == record['calc_date'],
                                        CorrelationData.time_window == record['time_window']
                                    )
                                ).first()
                                
                                if existing:
                                    if existing.correlation_value != record['correlation_value']:
                                        existing.correlation_value = record['correlation_value']
                                        session.commit()
                                        total_updated += 1
                                    else:
                                        total_ignored += 1
                                else:
                                    session.add(CorrelationData(**record))
                                    session.commit()
                                    total_inserted += 1
                                
                            except Exception as e:
                                session.rollback()
                                logger.error(f"Error procesando correlación para {pair} en {date}: {str(e)}")
                                total_ignored += 1
                            finally:
                                session.close()
                            
                        except Exception as e:
                            logger.error(f"Error procesando dato de correlación: {str(e)}")
                            total_ignored += 1
                
                processed_matrices += 1
        
        # Registrar en log
        self._log_process(
            process_name=f"load_correlation_json_{os.path.basename(file_path)}",
            status="success",
            records_processed=total_inserted + total_updated,
            execution_duration=None
        )
        
        return {
            'status': 'success',
            'file': file_path,
            'matrices_processed': processed_matrices,
            'inserted': total_inserted,
            'updated': total_updated,
            'ignored': total_ignored
        }
    
    def batch_process_directory(self, directory: str, file_pattern: str = '*.json') -> Dict:
        """
        Procesa todos los archivos de correlación en un directorio.
        
        Args:
            directory: Directorio con archivos
            file_pattern: Patrón para filtrar archivos
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        start_time = datetime.now()
        
        # Buscar archivos
        file_pattern_path = os.path.join(directory, file_pattern)
        files = glob.glob(file_pattern_path)
        
        if not files:
            logger.warning(f"No se encontraron archivos con patrón {file_pattern} en {directory}")
            return {
                'status': 'warning',
                'message': f'No se encontraron archivos con patrón {file_pattern}',
                'files_processed': 0,
                'results': {}
            }
        
        logger.info(f"Procesando {len(files)} archivos en {directory}")
        
        # Procesar cada archivo
        results = {}
        success_count = 0
        error_count = 0
        
        for file_path in files:
            try:
                result = self.load_correlation_data(file_path)
                
                # Usar nombre de archivo como clave
                filename = os.path.basename(file_path)
                results[filename] = result
                
                if result['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {str(e)}")
                filename = os.path.basename(file_path)
                results[filename] = {
                    'file': file_path,
                    'status': 'error',
                    'message': str(e)
                }
                error_count += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Registrar proceso en log
        self._log_process(
            process_name=f"batch_load_correlation_{os.path.basename(directory)}",
            status="success" if error_count == 0 else "partial_success",
            records_processed=len(files),
            execution_duration=duration
        )
        
        logger.info(f"Procesamiento por lotes completado: {success_count} éxitos, {error_count} errores, duración: {duration:.2f} segundos")
        
        return {
            'status': 'success' if error_count == 0 else 'partial_success',
            'files_processed': len(files),
            'success_count': success_count,
            'error_count': error_count,
            'duration_seconds': duration,
            'results': results
        }
    
    def _log_process(self, process_name: str, status: str,
                   error_message: Optional[str] = None,
                   records_processed: Optional[int] = None,
                   execution_duration: Optional[float] = None) -> None:
        """
        Registra una entrada en el log de procesos ETL.
        
        Args:
            process_name: Nombre del proceso
            status: Estado del proceso ('success', 'failed', etc.)
            error_message: Mensaje de error (opcional)
            records_processed: Número de registros procesados (opcional)
            execution_duration: Duración de la ejecución en segundos (opcional)
        """
        session = self.Session()
        try:
            log_entry = ETLLog(
                process_name=process_name,
                execution_time=datetime.now(),
                status=status,
                error_message=error_message,
                records_processed=records_processed,
                execution_duration=execution_duration
            )
            
            session.add(log_entry)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error registrando entrada de log: {str(e)}")
        finally:
            session.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cargar datos de correlaciones en la base de datos')
    parser.add_argument('--dir', type=str, required=True, help='Directorio con archivos de correlación')
    parser.add_argument('--pattern', type=str, default='*.json', help='Patrón para filtrar archivos')
    
    args = parser.parse_args()
    
    loader = CorrelationLoader()
    result = loader.batch_process_directory(
        directory=args.dir,
        file_pattern=args.pattern
    )
    
    print(f"Resultado: {result['status']}")
    print(f"Archivos procesados: {result['files_processed']}")
    print(f"Éxitos: {result['success_count']}")
    print(f"Errores: {result['error_count']}")
    print(f"Duración: {result['duration_seconds']:.2f} segundos")