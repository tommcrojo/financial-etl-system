#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para cargar datos de acciones procesados en la base de datos.
Implementa funciones para insertar y actualizar datos de precios e indicadores.
"""

import os
import sys
import glob
import logging
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
from src.database.create_database_schema import Asset, StockPrice, TechnicalIndicator, ETLLog, DB_URL

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/stock_data_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stock_data_loader')

# Cargar variables de entorno
load_dotenv()


class StockDataLoader:
    """
    Clase para cargar datos de acciones procesados en la base de datos.
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
    
    def register_asset(self, symbol: str, name: str, asset_type: str,
                      exchange: Optional[str] = None,
                      sector: Optional[str] = None,
                      industry: Optional[str] = None) -> int:
        """
        Registra un activo en la base de datos si no existe.
        
        Args:
            symbol: Símbolo del activo
            name: Nombre del activo
            asset_type: Tipo de activo ('stock', 'index', etc.)
            exchange: Bolsa donde cotiza (opcional)
            sector: Sector al que pertenece (opcional)
            industry: Industria a la que pertenece (opcional)
            
        Returns:
            ID del activo creado o existente
        """
        session = self.Session()
        try:
            # Verificar si el activo ya existe
            existing_asset = session.query(Asset).filter(Asset.symbol == symbol).first()
            
            if existing_asset:
                logger.info(f"Activo {symbol} ya existe con ID {existing_asset.asset_id}")
                
                # Actualizar información si es necesario
                updated = False
                
                if existing_asset.name != name:
                    existing_asset.name = name
                    updated = True
                
                if exchange and existing_asset.exchange != exchange:
                    existing_asset.exchange = exchange
                    updated = True
                
                if sector and existing_asset.sector != sector:
                    existing_asset.sector = sector
                    updated = True
                
                if industry and existing_asset.industry != industry:
                    existing_asset.industry = industry
                    updated = True
                
                if updated:
                    existing_asset.updated_at = datetime.now()
                    session.commit()
                    logger.info(f"Información actualizada para activo {symbol}")
                
                return existing_asset.asset_id
            
            # Crear nuevo activo
            new_asset = Asset(
                symbol=symbol,
                name=name,
                asset_type=asset_type,
                exchange=exchange,
                sector=sector,
                industry=industry
            )
            
            session.add(new_asset)
            session.commit()
            
            logger.info(f"Nuevo activo registrado: {symbol} (ID: {new_asset.asset_id})")
            return new_asset.asset_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error registrando activo {symbol}: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_stock_prices(self, asset_id: int, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Carga datos de precios de acciones en la base de datos.
        
        Args:
            asset_id: ID del activo
            df: DataFrame con datos de precios
            
        Returns:
            Tupla (registros_insertados, registros_actualizados, registros_ignorados)
        """
        if df.empty:
            logger.warning(f"DataFrame vacío, no hay datos para cargar para asset_id {asset_id}")
            return (0, 0, 0)
        
        # Verificar columnas requeridas
        required_cols = ['date', 'close']
        optional_cols = ['open', 'high', 'low', 'adjusted_close', 'volume']
        
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            logger.error(f"Faltan columnas requeridas: {missing_required}")
            return (0, 0, 0)
        
        # Asegurar que date sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Renombrar columnas para coincidir con el esquema de la base de datos
        col_mapping = {
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price'
        }
        
        for old_col, new_col in col_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Preparar datos para inserción
        records = []
        now = datetime.now()
        
        for _, row in df.iterrows():
            record = {
                'asset_id': asset_id,
                'trade_date': row['date'],
                'close_price': row['close_price'],
                'created_at': now
            }
            
            # Añadir columnas opcionales si existen
            for col in ['open_price', 'high_price', 'low_price', 'adjusted_close', 'volume']:
                if col in df.columns and not pd.isna(row.get(col)):
                    record[col] = row[col]
            
            records.append(record)
        
        # Cargar datos en la base de datos
        session = self.Session()
        inserted, updated, ignored = 0, 0, 0
        
        try:
            # Procesar por lotes para mayor eficiencia
            batch_size = 500
            total_records = len(records)
            
            for i in range(0, total_records, batch_size):
                batch = records[i:i+batch_size]
                
                # Para cada registro en el lote
                for record in batch:
                    try:
                        # Verificar si ya existe
                        existing = session.query(StockPrice).filter(
                            and_(
                                StockPrice.asset_id == record['asset_id'],
                                StockPrice.trade_date == record['trade_date']
                            )
                        ).first()
                        
                        if existing:
                            # Actualizar si los valores son diferentes
                            update_needed = False
                            
                            for key, value in record.items():
                                if key not in ['asset_id', 'trade_date', 'created_at']:
                                    existing_value = getattr(existing, key)
                                    if value != existing_value and not (pd.isna(value) and pd.isna(existing_value)):
                                        setattr(existing, key, value)
                                        update_needed = True
                            
                            if update_needed:
                                updated += 1
                            else:
                                ignored += 1
                        else:
                            # Insertar nuevo registro
                            session.add(StockPrice(**record))
                            inserted += 1
                        
                    except IntegrityError:
                        session.rollback()
                        ignored += 1
                        continue
                
                # Commit al final de cada lote
                session.commit()
                
                logger.info(f"Procesado lote {i//batch_size + 1}/{(total_records-1)//batch_size + 1}")
            
            logger.info(f"Carga completada para asset_id {asset_id}: {inserted} insertados, {updated} actualizados, {ignored} ignorados")
            return (inserted, updated, ignored)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cargando precios para asset_id {asset_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_technical_indicators(self, asset_id: int, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Carga indicadores técnicos en la base de datos.
        
        Args:
            asset_id: ID del activo
            df: DataFrame con indicadores técnicos
            
        Returns:
            Tupla (registros_insertados, registros_actualizados, registros_ignorados)
        """
        if df.empty:
            logger.warning(f"DataFrame vacío, no hay indicadores para cargar para asset_id {asset_id}")
            return (0, 0, 0)
        
        # Verificar columna de fecha
        if 'date' not in df.columns:
            logger.error("Falta columna 'date' en DataFrame")
            return (0, 0, 0)
        
        # Asegurar que date sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Identificar columnas de indicadores
        indicator_prefixes = [
            'sma_', 'ema_', 'rsi_', 'macd_', 'bb_', 'stoch_', 'adx_', 'atr_', 
            'obv', 'cci_', 'mfi_', 'roc_', 'williams_r_', 'ichimoku_'
        ]
        
        indicator_columns = []
        for col in df.columns:
            # Verificar si la columna es un indicador técnico
            is_indicator = any(col.startswith(prefix) for prefix in indicator_prefixes)
            
            # También considerar otros indicadores especiales
            special_indicators = ['daily_return', 'cumulative_return', 'volatility_20d', 'trend_label']
            
            if is_indicator or col in special_indicators:
                indicator_columns.append(col)
        
        if not indicator_columns:
            logger.warning(f"No se encontraron columnas de indicadores técnicos en el DataFrame")
            return (0, 0, 0)
        
        logger.info(f"Cargando {len(indicator_columns)} indicadores técnicos para asset_id {asset_id}")
        
        # Preparar registros para inserción
        records = []
        now = datetime.now()
        
        for _, row in df.iterrows():
            trade_date = row['date']
            
            for indicator in indicator_columns:
                # Extraer parámetros del nombre del indicador si es posible
                param1 = None
                param2 = None
                
                # Intentar extraer parámetros de nombres como 'sma_20', 'macd_12_26_9'
                parts = indicator.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    param1 = float(parts[1])
                    if len(parts) > 2 and parts[2].isdigit():
                        param2 = float(parts[2])
                
                value = row[indicator]
                if pd.isna(value):
                    continue
                
                records.append({
                    'asset_id': asset_id,
                    'calc_date': trade_date,
                    'indicator_type': indicator,
                    'value': value,
                    'parameter1': param1,
                    'parameter2': param2,
                    'created_at': now
                })
        
        # Cargar en la base de datos
        session = self.Session()
        inserted, updated, ignored = 0, 0, 0
        
        try:
            # Procesar por lotes
            batch_size = 1000
            total_records = len(records)
            
            for i in range(0, total_records, batch_size):
                batch = records[i:i+batch_size]
                
                # Para cada registro en el lote
                for record in batch:
                    try:
                        # Verificar si ya existe
                        existing = session.query(TechnicalIndicator).filter(
                            and_(
                                TechnicalIndicator.asset_id == record['asset_id'],
                                TechnicalIndicator.calc_date == record['calc_date'],
                                TechnicalIndicator.indicator_type == record['indicator_type']
                            )
                        ).first()
                        
                        if existing:
                            # Actualizar si el valor es diferente
                            if existing.value != record['value']:
                                existing.value = record['value']
                                if record.get('parameter1') is not None:
                                    existing.parameter1 = record['parameter1']
                                if record.get('parameter2') is not None:
                                    existing.parameter2 = record['parameter2']
                                updated += 1
                            else:
                                ignored += 1
                        else:
                            # Insertar nuevo registro
                            session.add(TechnicalIndicator(**record))
                            inserted += 1
                        
                    except IntegrityError:
                        session.rollback()
                        ignored += 1
                        continue
                
                # Commit al final de cada lote
                session.commit()
                
                if total_records > batch_size:
                    logger.info(f"Procesado lote {i//batch_size + 1}/{(total_records-1)//batch_size + 1}")
            
            logger.info(f"Carga de indicadores completada para asset_id {asset_id}: {inserted} insertados, {updated} actualizados, {ignored} ignorados")
            return (inserted, updated, ignored)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cargando indicadores para asset_id {asset_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def process_stock_file(self, file_path: str, asset_type: str = 'stock',
                         auto_register: bool = True) -> Dict:
        """
        Procesa un archivo de datos de acciones e inserta en la base de datos.
        
        Args:
            file_path: Ruta al archivo CSV
            asset_type: Tipo de activo ('stock', 'index', etc.)
            auto_register: Si True, registra automáticamente el activo si no existe
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        try:
            # Extraer símbolo del nombre de archivo
            filename = os.path.basename(file_path)
            symbol = filename.split('_')[0].upper()
            
            # Si es un índice, añadir el prefijo '^' si no lo tiene
            if asset_type == 'index' and not symbol.startswith('^'):
                symbol = f"^{symbol}"
            
            logger.info(f"Procesando archivo {filename} para símbolo {symbol}")
            
            # Cargar datos
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.warning(f"Archivo vacío: {file_path}")
                return {
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Archivo vacío'
                }
            
            # Verificar si el activo está registrado
            session = self.Session()
            asset = session.query(Asset).filter(Asset.symbol == symbol).first()
            session.close()
            
            asset_id = None
            
            if asset:
                asset_id = asset.asset_id
                logger.info(f"Activo {symbol} encontrado con ID {asset_id}")
            elif auto_register:
                # Construir nombre si no se especifica
                name = symbol
                if asset_type == 'index':
                    # Para índices, usar nombres comunes
                    index_names = {
                        '^GSPC': 'S&P 500',
                        '^DJI': 'Dow Jones Industrial Average',
                        '^IXIC': 'NASDAQ Composite',
                        '^NYA': 'NYSE Composite',
                        '^RUT': 'Russell 2000',
                        '^VIX': 'CBOE Volatility Index',
                        '^FTSE': 'FTSE 100',
                        '^N225': 'Nikkei 225',
                        '^HSI': 'Hang Seng Index',
                        '^GDAXI': 'DAX'
                    }
                    if symbol in index_names:
                        name = index_names[symbol]
                
                # Registrar activo
                asset_id = self.register_asset(
                    symbol=symbol,
                    name=name,
                    asset_type=asset_type
                )
                logger.info(f"Activo {symbol} registrado automáticamente con ID {asset_id}")
            else:
                logger.error(f"Activo {symbol} no encontrado y auto_register=False")
                return {
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Activo no encontrado'
                }
            
            # Cargar precios
            prices_inserted, prices_updated, prices_ignored = self.load_stock_prices(asset_id, df)
            
            # Determinar si contiene indicadores técnicos
            has_indicators = any(col.startswith(('sma_', 'ema_', 'rsi_', 'macd_')) for col in df.columns)
            
            indicators_result = (0, 0, 0)
            if has_indicators:
                # Cargar indicadores técnicos
                indicators_result = self.load_technical_indicators(asset_id, df)
            
            indicators_inserted, indicators_updated, indicators_ignored = indicators_result
            
            # Registrar en log
            self._log_process(
                process_name=f"load_stock_data_{symbol}",
                status="success",
                records_processed=prices_inserted + prices_updated + indicators_inserted + indicators_updated,
                execution_duration=None  # Se calcula automáticamente
            )
            
            return {
                'symbol': symbol,
                'asset_id': asset_id,
                'status': 'success',
                'prices': {
                    'inserted': prices_inserted,
                    'updated': prices_updated,
                    'ignored': prices_ignored
                },
                'indicators': {
                    'inserted': indicators_inserted,
                    'updated': indicators_updated,
                    'ignored': indicators_ignored
                }
            }
            
        except Exception as e:
            logger.error(f"Error procesando archivo {file_path}: {str(e)}")
            
            # Registrar error en log
            self._log_process(
                process_name=f"load_stock_data_{os.path.basename(file_path)}",
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
    
    def batch_process_directory(self, directory: str, file_pattern: str = '*_processed.csv',
                              asset_type: str = 'stock') -> Dict:
        """
        Procesa todos los archivos de datos de acciones en un directorio.
        
        Args:
            directory: Directorio con archivos CSV
            file_pattern: Patrón para filtrar archivos
            asset_type: Tipo de activo ('stock', 'index', etc.)
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        start_time = datetime.now()
        
        # Buscar archivos
        file_pattern = os.path.join(directory, file_pattern)
        files = glob.glob(file_pattern)
        
        if not files:
            logger.warning(f"No se encontraron archivos con patrón {file_pattern}")
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
                result = self.process_stock_file(file_path, asset_type)
                
                symbol = result.get('symbol', os.path.basename(file_path))
                results[symbol] = result
                
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
            process_name=f"batch_load_{asset_type}_{os.path.basename(directory)}",
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
    
    parser = argparse.ArgumentParser(description='Cargar datos de acciones en la base de datos')
    parser.add_argument('--dir', type=str, required=True, help='Directorio con archivos CSV')
    parser.add_argument('--pattern', type=str, default='*_processed.csv', help='Patrón para filtrar archivos')
    parser.add_argument('--type', type=str, default='stock', choices=['stock', 'index'], help='Tipo de activo')
    
    args = parser.parse_args()
    
    loader = StockDataLoader()
    result = loader.batch_process_directory(
        directory=args.dir,
        file_pattern=args.pattern,
        asset_type=args.type
    )
    
    print(f"Resultado: {result['status']}")
    print(f"Archivos procesados: {result['files_processed']}")
    print(f"Éxitos: {result['success_count']}")
    print(f"Errores: {result['error_count']}")
    print(f"Duración: {result['duration_seconds']:.2f} segundos")