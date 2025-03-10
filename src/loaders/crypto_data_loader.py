#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para cargar datos de criptomonedas procesados en la base de datos.
Implementa funciones para insertar y actualizar datos de mercado y de red.
"""

import os
import sys
import glob
import json
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
from src.database.create_database_schema import (
    Asset, CryptoMarketData, CryptoNetworkData, TechnicalIndicator, ETLLog, DB_URL
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/crypto_data_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('crypto_data_loader')

# Cargar variables de entorno
load_dotenv()


class CryptoDataLoader:
    """
    Clase para cargar datos de criptomonedas procesados en la base de datos.
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
    
    def register_asset(self, symbol: str, name: str, asset_type: str = 'crypto',
                      exchange: Optional[str] = None,
                      sector: Optional[str] = None,
                      industry: Optional[str] = None) -> int:
        """
        Registra un activo en la base de datos si no existe.
        
        Args:
            symbol: Símbolo del activo
            name: Nombre del activo
            asset_type: Tipo de activo (default: 'crypto')
            exchange: Bolsa donde cotiza (opcional)
            sector: Sector al que pertenece (opcional)
            industry: Industria a la que pertenece (opcional)
            
        Returns:
            ID del activo creado o existente
        """
        session = self.Session()
        try:
            # Asegurar que el símbolo esté en mayúsculas para criptomonedas
            symbol = symbol.upper()
            
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
    
    def load_crypto_market_data(self, asset_id: int, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Carga datos de mercado de criptomonedas en la base de datos.
        
        Args:
            asset_id: ID del activo
            df: DataFrame con datos de mercado
            
        Returns:
            Tupla (registros_insertados, registros_actualizados, registros_ignorados)
        """
        if df.empty:
            logger.warning(f"DataFrame vacío, no hay datos para cargar para asset_id {asset_id}")
            return (0, 0, 0)
        
        # Verificar columnas requeridas
        required_cols = ['date', 'price']
        optional_cols = ['market_cap', 'volume', 'circulating_supply', 'total_supply', 'price_change_24h']
        
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            # Intentar adaptar si hay columnas equivalentes (close en lugar de price)
            if 'price' in missing_required and 'close' in df.columns:
                df['price'] = df['close']
                missing_required.remove('price')
            
            if missing_required:
                logger.error(f"Faltan columnas requeridas: {missing_required}")
                return (0, 0, 0)
        
        # Asegurar que date sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Renombrar columnas para coincidir con el esquema de la base de datos
        col_mapping = {
            'price': 'price_usd',
            'volume': 'volume_24h'
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
                'price_usd': row['price_usd'],
                'created_at': now
            }
            
            # Añadir columnas opcionales si existen
            for col in ['market_cap', 'volume_24h', 'circulating_supply', 'total_supply', 'price_change_24h']:
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
                        existing = session.query(CryptoMarketData).filter(
                            and_(
                                CryptoMarketData.asset_id == record['asset_id'],
                                CryptoMarketData.trade_date == record['trade_date']
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
                            session.add(CryptoMarketData(**record))
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
            logger.error(f"Error cargando datos de mercado para asset_id {asset_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_crypto_network_data(self, asset_id: int, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Tuple[int, int, int]:
        """
        Carga datos de red de criptomonedas en la base de datos.
        
        Args:
            asset_id: ID del activo
            data: DataFrame o diccionario con datos de red
            
        Returns:
            Tupla (registros_insertados, registros_actualizados, registros_ignorados)
        """
        # Convertir dict a lista de dicts si es necesario
        if isinstance(data, dict) and not isinstance(data, pd.DataFrame):
            # Verificar si es un dict con varias fechas o un single dict
            if 'trade_date' in data or 'date' in data:
                data = [data]
            else:
                # Intentar extraer series temporales si están anidadas
                records = []
                if any(isinstance(data.get(k), (list, dict)) for k in data):
                    logger.warning(f"Formato de datos de red complejo, intentando extraer series temporales")
                    # Por ahora, solo tomamos los valores simples como último valor conocido
                    network_data = {
                        k: v for k, v in data.items() 
                        if not isinstance(v, (list, dict)) and not k.startswith('_')
                    }
                    if network_data:
                        records.append(network_data)
                else:
                    records.append(data)
                data = records
        
        # Si es DataFrame, convertir a lista de dicts
        if isinstance(data, pd.DataFrame):
            if data.empty:
                logger.warning(f"DataFrame vacío, no hay datos para cargar para asset_id {asset_id}")
                return (0, 0, 0)
            
            # Asegurar que date sea datetime
            if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
            
            # Convertir a lista de diccionarios
            data = data.to_dict(orient='records')
        
        # Si no hay datos, salir
        if not data:
            logger.warning(f"No hay datos de red para cargar para asset_id {asset_id}")
            return (0, 0, 0)
        
        logger.info(f"Cargando {len(data)} registros de datos de red para asset_id {asset_id}")
        
        # Mapeo de nombres de columnas
        field_mapping = {
            'date': 'trade_date',
            'hash_rate': 'hashrate',
            'hashRate': 'hashrate',
            'active_wallets': 'active_addresses',
            'active_accounts': 'active_addresses',
            'transaction_fee': 'avg_transaction_fee',
            'avg_fee': 'avg_transaction_fee',
            'transaction_value': 'avg_transaction_value',
            'avg_value': 'avg_transaction_value',
            'transactions': 'transaction_count',
            'tx_count': 'transaction_count'
        }
        
        # Preparar registros para inserción
        records = []
        now = datetime.now()
        
        for item in data:
            record = {
                'asset_id': asset_id,
                'created_at': now
            }
            
            # Extraer y mapear campos
            for src_field, dest_field in field_mapping.items():
                if src_field in item and item[src_field] is not None:
                    record[dest_field] = item[src_field]
            
            # También copiar campos con nombres correctos
            for field in ['trade_date', 'hashrate', 'active_addresses', 
                         'avg_transaction_fee', 'avg_transaction_value', 'transaction_count']:
                if field in item and item[field] is not None:
                    record[field] = item[field]
            
            # Verificar campo de fecha requerido
            if 'trade_date' not in record:
                # Sin fecha, no podemos insertar este registro
                logger.warning(f"Registro sin fecha encontrado, ignorando")
                continue
            
            # Convertir fecha si es string
            if isinstance(record['trade_date'], str):
                try:
                    record['trade_date'] = pd.to_datetime(record['trade_date'])
                except:
                    logger.warning(f"Error convirtiendo fecha: {record['trade_date']}")
                    continue
            
            records.append(record)
        
        if not records:
            logger.warning(f"No hay registros válidos para cargar después del procesamiento")
            return (0, 0, 0)
        
        # Cargar en la base de datos
        session = self.Session()
        inserted, updated, ignored = 0, 0, 0
        
        try:
            # Procesar todos los registros (normalmente son pocos para datos de red)
            for record in records:
                try:
                    # Verificar si ya existe
                    existing = session.query(CryptoNetworkData).filter(
                        and_(
                            CryptoNetworkData.asset_id == record['asset_id'],
                            CryptoNetworkData.trade_date == record['trade_date']
                        )
                    ).first()
                    
                    if existing:
                        # Actualizar si hay valores nuevos
                        update_needed = False
                        
                        for key, value in record.items():
                            if key not in ['asset_id', 'trade_date', 'created_at']:
                                if hasattr(existing, key):
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
                        session.add(CryptoNetworkData(**record))
                        inserted += 1
                    
                except IntegrityError:
                    session.rollback()
                    ignored += 1
                    continue
            
            # Commit al final
            session.commit()
            
            logger.info(f"Carga de datos de red completada para asset_id {asset_id}: {inserted} insertados, {updated} actualizados, {ignored} ignorados")
            return (inserted, updated, ignored)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cargando datos de red para asset_id {asset_id}: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_crypto_technical_indicators(self, asset_id: int, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Carga indicadores técnicos de criptomonedas en la base de datos.
        
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
            'obv', 'cci_', 'mfi_', 'roc_', 'williams_r_', 'ichimoku_', 'volatility_'
        ]
        
        special_indicators = [
            'trend', 'golden_cross', 'death_cross', 'log_return',
            'return_7d', 'return_30d', 'nvt_ratio', 'nvt_signal',
            'market_dominance', 'market_cycle'
        ]
        
        indicator_columns = []
        for col in df.columns:
            # Verificar si la columna es un indicador técnico
            is_indicator = (
                any(col.startswith(prefix) for prefix in indicator_prefixes) or
                any(indicator in col for indicator in special_indicators)
            )
            
            if is_indicator:
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
    
    def process_crypto_directory(self, crypto_dir: str, auto_register: bool = True) -> Dict:
        """
        Procesa un directorio de una criptomoneda e inserta datos en la base de datos.
        
        Args:
            crypto_dir: Ruta al directorio de una criptomoneda
            auto_register: Si True, registra automáticamente el activo si no existe
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        try:
            # Extraer símbolo del nombre del directorio
            dirname = os.path.basename(crypto_dir)
            symbol = dirname.upper()
            
            logger.info(f"Procesando directorio {dirname} para símbolo {symbol}")
            
            # Verificar si el activo está registrado
            session = self.Session()
            asset = session.query(Asset).filter(Asset.symbol == symbol).first()
            session.close()
            
            asset_id = None
            
            if asset:
                asset_id = asset.asset_id
                logger.info(f"Activo {symbol} encontrado con ID {asset_id}")
            elif auto_register:
                # Buscar archivo de información
                info_file = os.path.join(crypto_dir, f"{dirname}_info.json")
                info_file_clean = os.path.join(crypto_dir, f"{dirname}_info_clean.json")
                
                name = symbol  # Valor por defecto
                
                # Intentar obtener nombre desde archivo de información
                for info_path in [info_file_clean, info_file]:
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, 'r') as f:
                                info_data = json.load(f)
                                
                                if info_data and 'name' in info_data:
                                    name = info_data['name']
                        except:
                            pass
                
                # Registrar activo
                asset_id = self.register_asset(
                    symbol=symbol,
                    name=name,
                    asset_type='crypto'
                )
                logger.info(f"Activo {symbol} registrado automáticamente con ID {asset_id}")
            else:
                logger.error(f"Activo {symbol} no encontrado y auto_register=False")
                return {
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Activo no encontrado'
                }
            
            results = {
                'symbol': symbol,
                'asset_id': asset_id,
                'status': 'success',
                'market_data': {'inserted': 0, 'updated': 0, 'ignored': 0},
                'network_data': {'inserted': 0, 'updated': 0, 'ignored': 0},
                'indicators': {'inserted': 0, 'updated': 0, 'ignored': 0}
            }
            
            # 1. Procesar datos de mercado
            market_files_patterns = [
                f"{dirname}_market_cycles.csv",
                f"{dirname}_indicators.csv",
                f"{dirname}_market_clean.csv"
            ]
            
            for pattern in market_files_patterns:
                market_file = os.path.join(crypto_dir, pattern)
                if os.path.exists(market_file):
                    try:
                        df = pd.read_csv(market_file)
                        
                        if not df.empty:
                            inserted, updated, ignored = self.load_crypto_market_data(asset_id, df)
                            results['market_data'] = {
                                'inserted': inserted,
                                'updated': updated,
                                'ignored': ignored
                            }
                            
                            # Si este archivo tiene indicadores, cargarlos también
                            if pattern in [f"{dirname}_market_cycles.csv", f"{dirname}_indicators.csv"]:
                                indicators_inserted, indicators_updated, indicators_ignored = self.load_crypto_technical_indicators(asset_id, df)
                                results['indicators'] = {
                                    'inserted': indicators_inserted,
                                    'updated': indicators_updated,
                                    'ignored': indicators_ignored
                                }
                            
                            # Ya procesamos un archivo, no seguir con los demás
                            break
                    except Exception as e:
                        logger.error(f"Error procesando {market_file}: {str(e)}")
            
            # 2. Procesar datos de red
            network_file = os.path.join(crypto_dir, f"{dirname}_network_data.json")
            network_file_clean = os.path.join(crypto_dir, f"{dirname}_network_clean.json")
            
            for network_path in [network_file_clean, network_file]:
                if os.path.exists(network_path):
                    try:
                        with open(network_path, 'r') as f:
                            network_data = json.load(f)
                            
                            if network_data:
                                inserted, updated, ignored = self.load_crypto_network_data(asset_id, network_data)
                                results['network_data'] = {
                                    'inserted': inserted,
                                    'updated': updated,
                                    'ignored': ignored
                                }
                                
                                # Ya procesamos un archivo, no seguir con el otro
                                break
                    except Exception as e:
                        logger.error(f"Error procesando {network_path}: {str(e)}")
            
            # Registrar en log
            self._log_process(
                process_name=f"load_crypto_data_{symbol}",
                status="success",
                records_processed=(
                    results['market_data']['inserted'] + results['market_data']['updated'] +
                    results['network_data']['inserted'] + results['network_data']['updated'] +
                    results['indicators']['inserted'] + results['indicators']['updated']
                ),
                execution_duration=None  # Se calcula automáticamente
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error procesando directorio {crypto_dir}: {str(e)}")
            
            # Registrar error en log
            self._log_process(
                process_name=f"load_crypto_data_{os.path.basename(crypto_dir)}",
                status="failed",
                error_message=str(e),
                records_processed=0,
                execution_duration=None
            )
            
            return {
                'directory': crypto_dir,
                'status': 'error',
                'message': str(e)
            }
    
    def batch_process_directory(self, base_directory: str, auto_register: bool = True) -> Dict:
        """
        Procesa todos los subdirectorios de criptomonedas en un directorio base.
        
        Args:
            base_directory: Directorio base con subdirectorios por criptomoneda
            auto_register: Si True, registra automáticamente los activos que no existan
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        start_time = datetime.now()
        
        # Obtener lista de subdirectorios (uno por criptomoneda)
        subdirs = [d for d in os.listdir(base_directory) 
                  if os.path.isdir(os.path.join(base_directory, d)) and 
                     d not in ['.git', 'data', '__pycache__']]
        
        if not subdirs:
            logger.warning(f"No se encontraron subdirectorios en {base_directory}")
            return {
                'status': 'warning',
                'message': f'No se encontraron subdirectorios en {base_directory}',
                'directories_processed': 0,
                'results': {}
            }
        
        logger.info(f"Procesando {len(subdirs)} directorios en {base_directory}")
        
        # Procesar cada subdirectorio
        results = {}
        success_count = 0
        error_count = 0
        
        for subdir in subdirs:
            try:
                dir_path = os.path.join(base_directory, subdir)
                result = self.process_crypto_directory(dir_path, auto_register)
                
                symbol = result.get('symbol', subdir.upper())
                results[symbol] = result
                
                if result['status'] == 'success':
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error procesando {subdir}: {str(e)}")
                results[subdir.upper()] = {
                    'directory': os.path.join(base_directory, subdir),
                    'status': 'error',
                    'message': str(e)
                }
                error_count += 1
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Registrar proceso en log
        self._log_process(
            process_name=f"batch_load_crypto_{os.path.basename(base_directory)}",
            status="success" if error_count == 0 else "partial_success",
            records_processed=len(subdirs),
            execution_duration=duration
        )
        
        logger.info(f"Procesamiento por lotes completado: {success_count} éxitos, {error_count} errores, duración: {duration:.2f} segundos")
        
        return {
            'status': 'success' if error_count == 0 else 'partial_success',
            'directories_processed': len(subdirs),
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
    
    parser = argparse.ArgumentParser(description='Cargar datos de criptomonedas en la base de datos')
    parser.add_argument('--dir', type=str, required=True, help='Directorio base con subdirectorios por criptomoneda')
    parser.add_argument('--no-auto-register', action='store_true', help='No registrar activos automáticamente')
    
    args = parser.parse_args()
    
    loader = CryptoDataLoader()
    result = loader.batch_process_directory(
        base_directory=args.dir,
        auto_register=not args.no_auto_register
    )
    
    print(f"Resultado: {result['status']}")
    print(f"Directorios procesados: {result['directories_processed']}")
    print(f"Éxitos: {result['success_count']}")
    print(f"Errores: {result['error_count']}")
    print(f"Duración: {result['duration_seconds']:.2f} segundos")