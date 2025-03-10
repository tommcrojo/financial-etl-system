#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para transformación y procesamiento de datos empresariales.
Realiza limpieza, normalización y cálculo de ratios financieros.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/company_transformer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('company_transformer')


class CompanyDataTransformer:
    """
    Clase para transformar y procesar datos empresariales y financieros.
    """
    
    def __init__(self):
        """Inicializa el transformador de datos empresariales."""
        pass
    
    def clean_income_statement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y normaliza los datos del estado de resultados.
        
        Args:
            df: DataFrame con datos del estado de resultados
            
        Returns:
            DataFrame limpio y normalizado
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para limpiar")
            return df
        
        logger.info(f"Limpiando estado de resultados, shape inicial: {df.shape}")
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        # 1. Normalizar nombres de columnas
        if 'item' in data.columns:
            # Convertir nombres de todas las columnas a minúsculas excepto 'item'
            data.columns = ['item' if col.lower() == 'item' else col.lower() for col in data.columns]
            
            # Normalizar filas comunes del estado de resultados
            item_mapping = {
                # Mapeo para nombres comunes de filas de estado de resultados
                r'(?i)total.*revenue': 'total_revenue',
                r'(?i)revenue': 'revenue',
                r'(?i)sales': 'revenue',
                r'(?i)cost.*revenue': 'cost_of_revenue',
                r'(?i)cost.*goods.*sold': 'cost_of_revenue',
                r'(?i)cogs': 'cost_of_revenue',
                r'(?i)gross.*profit': 'gross_profit',
                r'(?i)operating.*expenses': 'operating_expenses',
                r'(?i)total.*operating.*expenses': 'total_operating_expenses',
                r'(?i)research.*development': 'rd_expense',
                r'(?i)r&d': 'rd_expense',
                r'(?i)selling.*general.*administrative': 'sga_expense',
                r'(?i)sga': 'sga_expense',
                r'(?i)operating.*income': 'operating_income',
                r'(?i)interest.*expense': 'interest_expense',
                r'(?i)income.*tax': 'income_tax',
                r'(?i)net.*income': 'net_income',
                r'(?i)earnings.*per.*share': 'eps',
                r'(?i)diluted.*eps': 'diluted_eps',
                r'(?i)ebitda': 'ebitda'
            }
            
            # Aplicar el mapeo usando expresiones regulares
            for pattern, replacement in item_mapping.items():
                data.loc[data['item'].str.match(pattern, na=False), 'item'] = replacement
        
        # 2. Convertir columnas de datos a numérico
        numeric_cols = [col for col in data.columns if col != 'item']
        
        for col in numeric_cols:
            # Si la columna ya es numérica, continuar
            if pd.api.types.is_numeric_dtype(data[col]):
                continue
                
            # Convertir strings a numérico
            try:
                # Manejar formatos comunes como "1,234.56", "$1,234.56", "(1,234.56)" para números negativos
                data[col] = data[col].astype(str)
                
                # Reemplazar paréntesis por signo negativo
                data[col] = data[col].str.replace(r'\((.*?)\)', r'-\1', regex=True)
                
                # Eliminar símbolos de moneda y comas
                data[col] = data[col].str.replace(r'[,$%]', '', regex=True)
                
                # Convertir a numérico
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
            except Exception as e:
                logger.error(f"Error convirtiendo columna '{col}' a numérico: {str(e)}")
        
        # 3. Manejar valores faltantes
        # Calcular porcentaje de valores nulos por columna
        null_pct = data[numeric_cols].isnull().mean() * 100
        
        # Eliminar columnas con más de 70% de valores nulos
        cols_to_drop = null_pct[null_pct > 70].index.tolist()
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            logger.info(f"Eliminadas columnas con demasiados valores nulos: {cols_to_drop}")
        
        # Para el resto de valores nulos, usar 0 (asumiendo que son valores no reportados)
        data = data.fillna(0)
        
        # 4. Verificar coherencia de los datos
        # Asegurar que gross_profit = revenue - cost_of_revenue
        if all(item in data['item'].values for item in ['revenue', 'cost_of_revenue', 'gross_profit']):
            for col in numeric_cols:
                if col == 'item':
                    continue
                
                rev_idx = data[data['item'] == 'revenue'].index[0]
                cost_idx = data[data['item'] == 'cost_of_revenue'].index[0]
                gp_idx = data[data['item'] == 'gross_profit'].index[0]
                
                # Verificar discrepancia
                expected_gp = data.loc[rev_idx, col] - data.loc[cost_idx, col]
                actual_gp = data.loc[gp_idx, col]
                
                # Si hay discrepancia significativa, ajustar
                if abs(expected_gp - actual_gp) > 0.1 * abs(expected_gp):
                    logger.warning(f"Discrepancia en gross_profit para columna '{col}'. Ajustando.")
                    data.loc[gp_idx, col] = expected_gp
        
        logger.info(f"Limpieza completada, shape final: {data.shape}")
        return data
    
    def clean_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y normaliza los datos del balance general.
        
        Args:
            df: DataFrame con datos del balance general
            
        Returns:
            DataFrame limpio y normalizado
        """
        if df.empty:
            logger.warning("DataFrame vacío, no hay datos para limpiar")
            return df
        
        logger.info(f"Limpiando balance general, shape inicial: {df.shape}")
        
        # Crear copia para no modificar el original
        data = df.copy()
        
        # 1. Normalizar nombres de columnas
        if 'item' in data.columns:
            # Convertir nombres de todas las columnas a minúsculas excepto 'item'
            data.columns = ['item' if col.lower() == 'item' else col.lower() for col in data.columns]
            
            # Normalizar filas comunes del balance general
            item_mapping = {
                # Activos
                r'(?i)total.*assets': 'total_assets',
                r'(?i)current.*assets': 'current_assets',
                r'(?i)cash.*equivalents': 'cash_and_equivalents',
                r'(?i)cash': 'cash_and_equivalents',
                r'(?i)short.*term.*investments': 'short_term_investments',
                r'(?i)accounts.*receivable': 'accounts_receivable',
                r'(?i)inventory': 'inventory',
                r'(?i)non.*current.*assets': 'non_current_assets',
                r'(?i)property.*plant.*equipment': 'ppe',
                r'(?i)ppe': 'ppe',
                r'(?i)long.*term.*investments': 'long_term_investments',
                r'(?i)goodwill': 'goodwill',
                r'(?i)intangible.*assets': 'intangible_assets',
                
                # Pasivos
                r'(?i)total.*liabilities': 'total_liabilities',
                r'(?i)current.*liabilities': 'current_liabilities',
                r'(?i)accounts.*payable': 'accounts_payable',
                r'(?i)short.*term.*debt': 'short_term_debt',
                r'(?i)non.*current.*liabilities': 'non_current_liabilities',
                r'(?i)long.*term.*debt': 'long_term_debt',
                
                # Patrimonio
                r'(?i)total.*equity': 'total_equity',
                r'(?i)stockholders.*equity': 'total_equity',
                r'(?i)common.*stock': 'common_stock',
                r'(?i)retained.*earnings': 'retained_earnings',
                r'(?i)treasury.*stock': 'treasury_stock'
            }
            
            # Aplicar el mapeo usando expresiones regulares
            for pattern, replacement in item_mapping.items():
                data.loc[data['item'].str.match(pattern, na=False), 'item'] = replacement
        
        # 2. Convertir columnas de datos a numérico
        numeric_cols = [col for col in data.columns if col != 'item']
        
        for col in numeric_cols:
            # Si la columna ya es numérica, continuar
            if pd.api.types.is_numeric_dtype(data[col]):
                continue
                
            # Convertir strings a numérico
            try:
                # Manejar formatos comunes como "1,234.56", "$1,234.56", "(1,234.56)" para números negativos
                data[col] = data[col].astype(str)
                
                # Reemplazar paréntesis por signo negativo
                data[col] = data[col].str.replace(r'\((.*?)\)', r'-\1', regex=True)
                
                # Eliminar símbolos de moneda y comas
                data[col] = data[col].str.replace(r'[,$%]', '', regex=True)
                
                # Convertir a numérico
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
            except Exception as e:
                logger.error(f"Error convirtiendo columna '{col}' a numérico: {str(e)}")
        
        # 3. Manejar valores faltantes
        # Calcular porcentaje de valores nulos por columna
        null_pct = data[numeric_cols].isnull().mean() * 100
        
        # Eliminar columnas con más de 70% de valores nulos
        cols_to_drop = null_pct[null_pct > 70].index.tolist()
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            logger.info(f"Eliminadas columnas con demasiados valores nulos: {cols_to_drop}")
        
        # Para el resto de valores nulos, usar 0 (asumiendo que son valores no reportados)
        data = data.fillna(0)
        
        # 4. Verificar coherencia de los datos
        # Verificar que total_assets = total_liabilities + total_equity
        if all(item in data['item'].values for item in ['total_assets', 'total_liabilities', 'total_equity']):
            for col in numeric_cols:
                if col == 'item':
                    continue
                
                assets_idx = data[data['item'] == 'total_assets'].index[0]
                liab_idx = data[data['item'] == 'total_liabilities'].index[0]
                equity_idx = data[data['item'] == 'total_equity'].index[0]
                
                # Verificar discrepancia
                assets = data.loc[assets_idx, col]
                liab_plus_equity = data.loc[liab_idx, col] + data.loc[equity_idx, col]
                
                # Si hay discrepancia significativa, ajustar
                if abs(assets - liab_plus_equity) > 0.1 * abs(assets):
                    logger.warning(f"Discrepancia en la ecuación contable para columna '{col}'. Ajustando.")
                    # Ajustar total_assets (asumiendo que los componentes son más precisos)
                    data.loc[assets_idx, col] = liab_plus_equity
        
        logger.info(f"Limpieza completada, shape final: {data.shape}")
        return data
    
    def normalize_financial_ratios(self, ratios_data: Dict) -> Dict:
        """
        Normaliza y limpia los ratios financieros.
        
        Args:
            ratios_data: Diccionario con ratios financieros
            
        Returns:
            Diccionario limpio y normalizado
        """
        if not ratios_data:
            logger.warning("Diccionario de ratios vacío")
            return {}
        
        logger.info("Normalizando ratios financieros")
        
        normalized_ratios = {}
        
        # Procesar cada sección de ratios
        for section_name, section_data in ratios_data.items():
            normalized_section = {}
            
            for ratio_name, ratio_values in section_data.items():
                normalized_values = {}
                
                for year, value in ratio_values.items():
                    # Convertir a numérico si es posible
                    try:
                        if value is None or value == 'N/A' or value == '-':
                            normalized_values[year] = None
                        else:
                            # Si es string, intentar convertir a numérico
                            if isinstance(value, str):
                                # Reemplazar paréntesis por signo negativo
                                value = re.sub(r'\((.*?)\)', r'-\1', value)
                                # Eliminar símbolos no numéricos
                                value = re.sub(r'[,$%]', '', value)
                            
                            normalized_values[year] = float(value)
                    except (ValueError, TypeError):
                        normalized_values[year] = None
                        logger.warning(f"No se pudo convertir valor '{value}' a numérico para ratio '{ratio_name}'")
                
                # Eliminar ratio si todos los valores son None
                if all(v is None for v in normalized_values.values()):
                    logger.warning(f"Eliminado ratio '{ratio_name}' por falta de datos")
                    continue
                
                normalized_section[ratio_name] = normalized_values
            
            # Añadir sección si tiene datos
            if normalized_section:
                normalized_ratios[section_name] = normalized_section
        
        logger.info("Normalización de ratios completada")
        return normalized_ratios
    
    def calculate_financial_ratios(self, income_df: pd.DataFrame, balance_df: pd.DataFrame,
                                 time_periods: Optional[List[str]] = None) -> Dict:
        """
        Calcula ratios financieros a partir de estados financieros.
        
        Args:
            income_df: DataFrame con estado de resultados limpio
            balance_df: DataFrame con balance general limpio
            time_periods: Lista de períodos/columnas para calcular ratios
            
        Returns:
            Diccionario con ratios financieros calculados
        """
        if income_df.empty or balance_df.empty:
            logger.warning("DataFrames vacíos, no se pueden calcular ratios")
            return {}
        
        logger.info("Calculando ratios financieros a partir de estados financieros")
        
        # Determinar las columnas (períodos) comunes en ambos dataframes
        income_cols = [col for col in income_df.columns if col != 'item']
        balance_cols = [col for col in balance_df.columns if col != 'item']
        
        if time_periods:
            # Usar los períodos especificados (si existen en ambos dataframes)
            common_periods = [col for col in time_periods if col in income_cols and col in balance_cols]
        else:
            # Usar intersección de columnas si no se especifican períodos
            common_periods = list(set(income_cols).intersection(set(balance_cols)))
        
        if not common_periods:
            logger.error("No hay períodos comunes entre estado de resultados y balance general")
            return {}
        
        logger.info(f"Calculando ratios para {len(common_periods)} períodos: {common_periods}")
        
        # Inicializar diccionario de ratios
        ratios = {
            "Profitability_Ratios": {},
            "Liquidity_Ratios": {},
            "Efficiency_Ratios": {},
            "Solvency_Ratios": {},
            "Valuation_Ratios": {}
        }
        
        # Funciones helper para obtener valores de los dataframes
        def get_income_value(item_name: str, period: str) -> float:
            """Obtiene un valor del estado de resultados"""
            if item_name not in income_df['item'].values:
                return 0
            return income_df.loc[income_df['item'] == item_name, period].iloc[0]
        
        def get_balance_value(item_name: str, period: str) -> float:
            """Obtiene un valor del balance general"""
            if item_name not in balance_df['item'].values:
                return 0
            return balance_df.loc[balance_df['item'] == item_name, period].iloc[0]
        
        # Calcular ratios para cada período
        for period in common_periods:
            # 1. Profitability Ratios
            
            # Gross Margin
            revenue = get_income_value('revenue', period) or get_income_value('total_revenue', period)
            cost_of_revenue = get_income_value('cost_of_revenue', period)
            
            if revenue and revenue != 0:
                gross_margin = ((revenue - cost_of_revenue) / revenue) * 100
                if 'Gross_Margin' not in ratios['Profitability_Ratios']:
                    ratios['Profitability_Ratios']['Gross_Margin'] = {}
                ratios['Profitability_Ratios']['Gross_Margin'][period] = gross_margin
            
            # Operating Margin
            operating_income = get_income_value('operating_income', period)
            
            if revenue and revenue != 0:
                operating_margin = (operating_income / revenue) * 100
                if 'Operating_Margin' not in ratios['Profitability_Ratios']:
                    ratios['Profitability_Ratios']['Operating_Margin'] = {}
                ratios['Profitability_Ratios']['Operating_Margin'][period] = operating_margin
            
            # Net Margin
            net_income = get_income_value('net_income', period)
            
            if revenue and revenue != 0:
                net_margin = (net_income / revenue) * 100
                if 'Net_Margin' not in ratios['Profitability_Ratios']:
                    ratios['Profitability_Ratios']['Net_Margin'] = {}
                ratios['Profitability_Ratios']['Net_Margin'][period] = net_margin
            
            # ROA (Return on Assets)
            total_assets = get_balance_value('total_assets', period)
            
            if total_assets and total_assets != 0:
                roa = (net_income / total_assets) * 100
                if 'ROA' not in ratios['Profitability_Ratios']:
                    ratios['Profitability_Ratios']['ROA'] = {}
                ratios['Profitability_Ratios']['ROA'][period] = roa
            
            # ROE (Return on Equity)
            total_equity = get_balance_value('total_equity', period)
            
            if total_equity and total_equity != 0:
                roe = (net_income / total_equity) * 100
                if 'ROE' not in ratios['Profitability_Ratios']:
                    ratios['Profitability_Ratios']['ROE'] = {}
                ratios['Profitability_Ratios']['ROE'][period] = roe
            
            # 2. Liquidity Ratios
            
            # Current Ratio
            current_assets = get_balance_value('current_assets', period)
            current_liabilities = get_balance_value('current_liabilities', period)
            
            if current_liabilities and current_liabilities != 0:
                current_ratio = current_assets / current_liabilities
                if 'Current_Ratio' not in ratios['Liquidity_Ratios']:
                    ratios['Liquidity_Ratios']['Current_Ratio'] = {}
                ratios['Liquidity_Ratios']['Current_Ratio'][period] = current_ratio
            
            # Quick Ratio
            inventory = get_balance_value('inventory', period)
            quick_assets = current_assets - inventory
            
            if current_liabilities and current_liabilities != 0:
                quick_ratio = quick_assets / current_liabilities
                if 'Quick_Ratio' not in ratios['Liquidity_Ratios']:
                    ratios['Liquidity_Ratios']['Quick_Ratio'] = {}
                ratios['Liquidity_Ratios']['Quick_Ratio'][period] = quick_ratio
            
            # Cash Ratio
            cash = get_balance_value('cash_and_equivalents', period)
            
            if current_liabilities and current_liabilities != 0:
                cash_ratio = cash / current_liabilities
                if 'Cash_Ratio' not in ratios['Liquidity_Ratios']:
                    ratios['Liquidity_Ratios']['Cash_Ratio'] = {}
                ratios['Liquidity_Ratios']['Cash_Ratio'][period] = cash_ratio
            
            # 3. Efficiency Ratios
            
            # Asset Turnover
            if total_assets and total_assets != 0:
                asset_turnover = revenue / total_assets
                if 'Asset_Turnover' not in ratios['Efficiency_Ratios']:
                    ratios['Efficiency_Ratios']['Asset_Turnover'] = {}
                ratios['Efficiency_Ratios']['Asset_Turnover'][period] = asset_turnover
            
            # Inventory Turnover
            if inventory and inventory != 0:
                inventory_turnover = cost_of_revenue / inventory
                if 'Inventory_Turnover' not in ratios['Efficiency_Ratios']:
                    ratios['Efficiency_Ratios']['Inventory_Turnover'] = {}
                ratios['Efficiency_Ratios']['Inventory_Turnover'][period] = inventory_turnover
            
            # Accounts Receivable Turnover
            accounts_receivable = get_balance_value('accounts_receivable', period)
            
            if accounts_receivable and accounts_receivable != 0:
                ar_turnover = revenue / accounts_receivable
                if 'Receivables_Turnover' not in ratios['Efficiency_Ratios']:
                    ratios['Efficiency_Ratios']['Receivables_Turnover'] = {}
                ratios['Efficiency_Ratios']['Receivables_Turnover'][period] = ar_turnover
            
            # 4. Solvency Ratios
            
            # Debt to Equity
            total_debt = (get_balance_value('short_term_debt', period) or 0) + \
                         (get_balance_value('long_term_debt', period) or 0)
            
            if total_equity and total_equity != 0:
                debt_to_equity = total_debt / total_equity
                if 'Debt_to_Equity' not in ratios['Solvency_Ratios']:
                    ratios['Solvency_Ratios']['Debt_to_Equity'] = {}
                ratios['Solvency_Ratios']['Debt_to_Equity'][period] = debt_to_equity
            
            # Debt Ratio
            if total_assets and total_assets != 0:
                debt_ratio = total_debt / total_assets
                if 'Debt_Ratio' not in ratios['Solvency_Ratios']:
                    ratios['Solvency_Ratios']['Debt_Ratio'] = {}
                ratios['Solvency_Ratios']['Debt_Ratio'][period] = debt_ratio
            
            # Interest Coverage
            interest_expense = get_income_value('interest_expense', period)
            
            if interest_expense and interest_expense != 0:
                interest_coverage = operating_income / abs(interest_expense)
                if 'Interest_Coverage' not in ratios['Solvency_Ratios']:
                    ratios['Solvency_Ratios']['Interest_Coverage'] = {}
                ratios['Solvency_Ratios']['Interest_Coverage'][period] = interest_coverage
            
            # 5. Valuation Ratios
            # Estos ratios generalmente requieren datos de mercado (precio de acción)
            # que no están disponibles en los estados financieros
            
            # P/E, P/B, EV/EBITDA, etc. requerirían datos adicionales
        
        logger.info("Cálculo de ratios financieros completado")
        return ratios
    
    def process_company_batch(self, input_dir: str, output_dir: str,
                             calculate_additional_ratios: bool = True) -> Dict[str, Dict]:
        """
        Procesa un lote de archivos de datos empresariales.
        
        Args:
            input_dir: Directorio de entrada con datos empresariales
            output_dir: Directorio de salida para datos procesados
            calculate_additional_ratios: Si True, calcula ratios adicionales
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        try:
            # Listar subdirectorios (uno por empresa)
            company_dirs = [d for d in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, d)) and d != '.git']
            
            if not company_dirs:
                logger.warning(f"No se encontraron directorios de empresas en {input_dir}")
                return {"error": "No se encontraron datos para procesar"}
            
            logger.info(f"Procesando datos de {len(company_dirs)} empresas")
            
            for company_dir in company_dirs:
                company_results = {
                    'income_statement': False,
                    'balance_sheet': False,
                    'ratios': False,
                    'additional_ratios': False
                }
                
                input_company_dir = os.path.join(input_dir, company_dir)
                output_company_dir = os.path.join(output_dir, company_dir)
                os.makedirs(output_company_dir, exist_ok=True)
                
                try:
                    # Procesar Estado de Resultados
                    income_file = f"{company_dir}_income_statement.csv"
                    income_path = os.path.join(input_company_dir, income_file)
                    
                    income_df = None
                    if os.path.exists(income_path):
                        income_df = pd.read_csv(income_path)
                        if not income_df.empty:
                            income_df_clean = self.clean_income_statement(income_df)
                            output_income_path = os.path.join(output_company_dir, f"{company_dir}_income_clean.csv")
                            income_df_clean.to_csv(output_income_path, index=False)
                            logger.info(f"Estado de resultados procesado guardado en {output_income_path}")
                            company_results['income_statement'] = True
                        else:
                            logger.warning(f"Archivo de estado de resultados vacío para {company_dir}")
                    else:
                        logger.warning(f"No se encontró archivo de estado de resultados para {company_dir}")
                    
                    # Procesar Balance General
                    balance_file = f"{company_dir}_balance_sheet.csv"
                    balance_path = os.path.join(input_company_dir, balance_file)
                    
                    balance_df = None
                    if os.path.exists(balance_path):
                        balance_df = pd.read_csv(balance_path)
                        if not balance_df.empty:
                            balance_df_clean = self.clean_balance_sheet(balance_df)
                            output_balance_path = os.path.join(output_company_dir, f"{company_dir}_balance_clean.csv")
                            balance_df_clean.to_csv(output_balance_path, index=False)
                            logger.info(f"Balance general procesado guardado en {output_balance_path}")
                            company_results['balance_sheet'] = True
                        else:
                            logger.warning(f"Archivo de balance general vacío para {company_dir}")
                    else:
                        logger.warning(f"No se encontró archivo de balance general para {company_dir}")
                    
                    # Procesar Ratios Financieros
                    ratios_file = f"{company_dir}_ratios.json"
                    ratios_path = os.path.join(input_company_dir, ratios_file)
                    
                    if os.path.exists(ratios_path):
                        with open(ratios_path, 'r') as f:
                            ratios_data = json.load(f)
                        
                        if ratios_data:
                            normalized_ratios = self.normalize_financial_ratios(ratios_data)
                            output_ratios_path = os.path.join(output_company_dir, f"{company_dir}_ratios_clean.json")
                            with open(output_ratios_path, 'w') as f:
                                json.dump(normalized_ratios, f, indent=2)
                            logger.info(f"Ratios normalizados guardados en {output_ratios_path}")
                            company_results['ratios'] = True
                        else:
                            logger.warning(f"Archivo de ratios vacío para {company_dir}")
                    else:
                        logger.warning(f"No se encontró archivo de ratios para {company_dir}")
                    
                    # Calcular ratios adicionales
                    if calculate_additional_ratios and income_df is not None and balance_df is not None:
                        if company_results['income_statement'] and company_results['balance_sheet']:
                            additional_ratios = self.calculate_financial_ratios(
                                income_df_clean, balance_df_clean
                            )
                            
                            if additional_ratios:
                                output_add_ratios_path = os.path.join(
                                    output_company_dir, f"{company_dir}_additional_ratios.json"
                                )
                                with open(output_add_ratios_path, 'w') as f:
                                    json.dump(additional_ratios, f, indent=2)
                                logger.info(f"Ratios adicionales guardados en {output_add_ratios_path}")
                                company_results['additional_ratios'] = True
                            else:
                                logger.warning(f"No se pudieron calcular ratios adicionales para {company_dir}")
                    
                except Exception as e:
                    logger.error(f"Error procesando empresa {company_dir}: {str(e)}")
                
                # Guardar resultados para esta empresa
                results[company_dir] = company_results
            
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
    transformer = CompanyDataTransformer()
    
    # Procesar datos de empresas
    results = transformer.process_company_batch(
        input_dir="data/companies/raw",
        output_dir="data/companies/processed",
        calculate_additional_ratios=True
    )
    
    print("Resultados del procesamiento:")
    for company, result in results.items():
        print(f"{company}: {result}")