#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo extractor de datos empresariales mediante web scraping.
Obtiene datos financieros y ratios de empresas desde fuentes públicas.
"""

import os
import time
import json
import random
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException
)
from webdriver_manager.chrome import ChromeDriverManager

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/company_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('company_scraper')

# Constantes
MAX_RETRIES = 3
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]
BASE_DELAY = 5  # Segundos base entre solicitudes


class CompanyDataScraper:
    """
    Extractor de datos empresariales mediante web scraping de sitios financieros públicos.
    """
    
    def __init__(self, use_selenium: bool = True, headless: bool = True):
        """
        Inicializa el scraper.
        
        Args:
            use_selenium: Si True, utiliza Selenium para sitios con JavaScript
            headless: Si True, ejecuta el navegador en modo headless (sin interfaz gráfica)
        """
        self.use_selenium = use_selenium
        self.headless = headless
        self.driver = None
        
        if use_selenium:
            self._init_selenium(headless)
    
    def _init_selenium(self, headless: bool = True) -> None:
        """
        Inicializa el driver de Selenium.
        
        Args:
            headless: Si True, ejecuta Chrome en modo headless
        """
        try:
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
            
            # Instalación automática del driver
            service = Service(ChromeDriverManager().install())
            
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("Selenium WebDriver inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar Selenium: {str(e)}")
            if self.driver:
                self.driver.quit()
            raise
    
    def __del__(self):
        """Cierra el driver de Selenium al finalizar"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Selenium WebDriver cerrado correctamente")
            except:
                pass
    
    def _random_delay(self) -> None:
        """
        Introduce un retraso aleatorio para evitar detección de scraping.
        """
        delay = BASE_DELAY + random.uniform(1, 5)
        logger.debug(f"Esperando {delay:.2f} segundos...")
        time.sleep(delay)
    
    def _get_with_requests(self, url: str) -> str:
        """
        Obtiene el contenido HTML de una URL usando requests.
        
        Args:
            url: URL a obtener
            
        Returns:
            Contenido HTML de la página
        """
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
            'Referer': 'https://www.google.com/'
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.text
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BASE_DELAY * (attempt + 1))  # Backoff exponencial
                else:
                    logger.error(f"Error al obtener {url} después de {MAX_RETRIES} intentos.")
                    raise
        
        return ""  # Nunca debería llegar aquí debido al raise anterior
    
    def _get_with_selenium(self, url: str, wait_for_selector: Optional[str] = None,
                          timeout: int = 20) -> str:
        """
        Obtiene el contenido HTML de una URL usando Selenium.
        
        Args:
            url: URL a obtener
            wait_for_selector: Selector CSS a esperar (opcional)
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            Contenido HTML de la página
        """
        if not self.driver:
            self._init_selenium(self.headless)
        
        for attempt in range(MAX_RETRIES):
            try:
                self.driver.get(url)
                
                if wait_for_selector:
                    try:
                        WebDriverWait(self.driver, timeout).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                        )
                    except TimeoutException:
                        logger.warning(f"Timeout esperando selector '{wait_for_selector}'")
                
                # Scroll para cargar contenido
                last_height = self.driver.execute_script("return document.body.scrollHeight")
                while True:
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    new_height = self.driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                
                return self.driver.page_source
                
            except (WebDriverException, Exception) as e:
                logger.error(f"Intento {attempt+1}/{MAX_RETRIES} fallido: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BASE_DELAY * (attempt + 1))
                    if isinstance(e, WebDriverException):
                        # Reinicializar el driver si hay problemas
                        try:
                            self.driver.quit()
                        except:
                            pass
                        self._init_selenium(self.headless)
                else:
                    logger.error(f"Error al obtener {url} con Selenium después de {MAX_RETRIES} intentos.")
                    raise
        
        return ""
    
    def _edgar_search_cik(self, ticker: str) -> str:
        """
        Busca el CIK (Central Index Key) de una empresa en EDGAR.
        
        Args:
            ticker: Símbolo de la acción
            
        Returns:
            CIK de la empresa o cadena vacía si no se encuentra
        """
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&Find=Search&owner=exclude&action=getcompany"
        
        try:
            html = self._get_with_requests(url)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Buscar el CIK en la página
            cik_info = soup.find('span', {'class': 'companyName'})
            if cik_info:
                # Formato típico: "APPLE INC (0000320193)"
                cik_text = cik_info.text
                if "(" in cik_text and ")" in cik_text:
                    cik = cik_text.split("(")[1].split(")")[0].strip()
                    # Eliminar ceros a la izquierda
                    cik = cik.lstrip("0")
                    return cik
            
            logger.warning(f"No se pudo encontrar CIK para {ticker}")
            return ""
            
        except Exception as e:
            logger.error(f"Error buscando CIK para {ticker}: {str(e)}")
            return ""
    
    def scrape_edgar_filings(self, ticker: str, form_type: str = "10-K",
                            limit: int = 5) -> List[Dict]:
        """
        Obtiene los últimos archivos presentados a la SEC para una empresa.
        
        Args:
            ticker: Símbolo de la acción
            form_type: Tipo de formulario ('10-K', '10-Q', etc.)
            limit: Número máximo de archivos a obtener
            
        Returns:
            Lista de diccionarios con información de los archivos
        """
        logger.info(f"Obteniendo archivos {form_type} para {ticker} desde EDGAR...")
        
        cik = self._edgar_search_cik(ticker)
        if not cik:
            logger.error(f"No se pudo obtener CIK para {ticker}")
            return []
        
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=exclude&count={limit}"
        
        try:
            self._random_delay()
            html = self._get_with_requests(url)
            soup = BeautifulSoup(html, 'html.parser')
            
            filings = []
            # Buscar tabla de archivos
            table = soup.find('table', {'class': 'tableFile2'})
            if not table:
                logger.warning(f"No se encontró tabla de archivos para {ticker}")
                return []
            
            rows = table.find_all('tr')
            for row in rows[1:]:  # Saltar encabezado
                cols = row.find_all('td')
                if len(cols) >= 4:
                    filing_type = cols[0].text.strip()
                    filing_date = cols[3].text.strip()
                    
                    # Obtener enlace al documento
                    filing_link = ""
                    documents_link = cols[1].find('a', {'id': 'documentsbutton'})
                    if documents_link and 'href' in documents_link.attrs:
                        documents_href = documents_link['href']
                        filing_link = f"https://www.sec.gov{documents_href}"
                    
                    filings.append({
                        'filing_type': filing_type,
                        'filing_date': filing_date,
                        'filing_link': filing_link
                    })
                    
                    if len(filings) >= limit:
                        break
            
            logger.info(f"Se encontraron {len(filings)} archivos {form_type} para {ticker}")
            return filings
            
        except Exception as e:
            logger.error(f"Error obteniendo archivos para {ticker}: {str(e)}")
            return []
    
    def scrape_income_statement_investing(self, ticker: str, 
                                         country: str = "us") -> pd.DataFrame:
        """
        Obtiene el estado de resultados desde Investing.com
        
        Args:
            ticker: Símbolo de la acción
            country: Código de país ('us', 'uk', etc.)
            
        Returns:
            DataFrame con el estado de resultados
        """
        logger.info(f"Obteniendo estado de resultados para {ticker} desde Investing.com...")
        
        # Primero, buscar la página de la empresa
        search_url = f"https://www.investing.com/search/?q={ticker}"
        
        try:
            self._random_delay()
            if self.use_selenium:
                html = self._get_with_selenium(search_url, ".js-inner-all-results")
            else:
                html = self._get_with_requests(search_url)
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Buscar el enlace a la página de la empresa
            company_link = None
            search_results = soup.select(".js-inner-all-results .js-inner-all-results-quote")
            
            for result in search_results:
                if result.get("data-country-id") == country.upper() and "EQUITIES" in result.get("data-type", ""):
                    link = result.select_one("a.second")
                    if link and 'href' in link.attrs:
                        company_link = link['href']
                        break
            
            if not company_link:
                logger.warning(f"No se encontró enlace para {ticker} en Investing.com")
                return pd.DataFrame()
            
            # Construir enlace a la página de financials
            if not company_link.startswith("http"):
                company_link = f"https://www.investing.com{company_link}"
            
            # Reemplazar '-profile' con '-financial-summary' para ir a los estados financieros
            financial_url = company_link.replace("-profile", "-financial-summary")
            
            # Obtener la página de financials
            self._random_delay()
            if self.use_selenium:
                html = self._get_with_selenium(financial_url, "#rrtable")
            else:
                html = self._get_with_requests(financial_url)
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Buscar la tabla de estado de resultados
            tables = soup.select("table.genTbl.reportTbl")
            if not tables:
                logger.warning(f"No se encontraron tablas financieras para {ticker}")
                return pd.DataFrame()
            
            income_statement = []
            
            # Procesar la primera tabla (normalmente es el estado de resultados)
            rows = tables[0].select("tbody tr")
            headers = [th.text.strip() for th in tables[0].select("thead tr th")]
            
            # Eliminar la primera columna si es "Period"
            if headers and headers[0].lower() in ["period", "período"]:
                columns = headers[1:]
            else:
                columns = headers
            
            # Procesar filas
            for row in rows:
                cells = row.select("td")
                if len(cells) >= len(headers):
                    row_item = {
                        'item': cells[0].text.strip()
                    }
                    
                    # Agregar valores por período
                    for i, col in enumerate(columns):
                        value = cells[i+1].text.strip()
                        # Convertir a numérico si es posible
                        try:
                            # Limpiar formato de números
                            value = value.replace(',', '')
                            value = float(value)
                        except:
                            pass
                        
                        row_item[col] = value
                    
                    income_statement.append(row_item)
            
            # Convertir a DataFrame
            df = pd.DataFrame(income_statement)
            logger.info(f"Estado de resultados obtenido exitosamente para {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de resultados para {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def scrape_balance_sheet_investing(self, ticker: str, 
                                      country: str = "us") -> pd.DataFrame:
        """
        Obtiene el balance general desde Investing.com
        
        Args:
            ticker: Símbolo de la acción
            country: Código de país ('us', 'uk', etc.)
            
        Returns:
            DataFrame con el balance general
        """
        logger.info(f"Obteniendo balance general para {ticker} desde Investing.com...")
        
        # La estructura es similar a scrape_income_statement_investing pero cambia la URL
        # Primero, buscar la página de la empresa
        search_url = f"https://www.investing.com/search/?q={ticker}"
        
        try:
            self._random_delay()
            if self.use_selenium:
                html = self._get_with_selenium(search_url, ".js-inner-all-results")
            else:
                html = self._get_with_requests(search_url)
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Buscar el enlace a la página de la empresa
            company_link = None
            search_results = soup.select(".js-inner-all-results .js-inner-all-results-quote")
            
            for result in search_results:
                if result.get("data-country-id") == country.upper() and "EQUITIES" in result.get("data-type", ""):
                    link = result.select_one("a.second")
                    if link and 'href' in link.attrs:
                        company_link = link['href']
                        break
            
            if not company_link:
                logger.warning(f"No se encontró enlace para {ticker} en Investing.com")
                return pd.DataFrame()
            
            # Construir enlace a la página de balance sheet
            if not company_link.startswith("http"):
                company_link = f"https://www.investing.com{company_link}"
            
            # Reemplazar '-profile' con '-balance-sheet' para ir al balance general
            balance_url = company_link.replace("-profile", "-balance-sheet")
            
            # Obtener la página de balance
            self._random_delay()
            if self.use_selenium:
                html = self._get_with_selenium(balance_url, "#rrtable")
            else:
                html = self._get_with_requests(balance_url)
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Buscar la tabla de balance general
            tables = soup.select("table.genTbl.reportTbl")
            if not tables:
                logger.warning(f"No se encontraron tablas de balance para {ticker}")
                return pd.DataFrame()
            
            balance_sheet = []
            
            # Procesar la tabla
            rows = tables[0].select("tbody tr")
            headers = [th.text.strip() for th in tables[0].select("thead tr th")]
            
            # Eliminar la primera columna si es descriptiva
            if headers and headers[0].lower() in ["period", "período", "item", "cuenta"]:
                columns = headers[1:]
            else:
                columns = headers
            
            # Procesar filas
            for row in rows:
                cells = row.select("td")
                if len(cells) >= len(headers):
                    row_item = {
                        'item': cells[0].text.strip()
                    }
                    
                    # Agregar valores por período
                    for i, col in enumerate(columns):
                        value = cells[i+1].text.strip()
                        # Convertir a numérico si es posible
                        try:
                            value = value.replace(',', '')
                            value = float(value)
                        except:
                            pass
                        
                        row_item[col] = value
                    
                    balance_sheet.append(row_item)
            
            # Convertir a DataFrame
            df = pd.DataFrame(balance_sheet)
            logger.info(f"Balance general obtenido exitosamente para {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo balance general para {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def scrape_ratios_marketwatch(self, ticker: str) -> Dict:
        """
        Obtiene los ratios financieros desde MarketWatch.
        
        Args:
            ticker: Símbolo de la acción
            
        Returns:
            Diccionario con los ratios financieros
        """
        logger.info(f"Obteniendo ratios financieros para {ticker} desde MarketWatch...")
        
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}/financials/ratios"
        
        try:
            self._random_delay()
            html = self._get_with_requests(url)
            soup = BeautifulSoup(html, 'html.parser')
            
            ratios = {}
            
            # Buscar secciones de ratios
            sections = soup.select(".table__section")
            for section in sections:
                section_title = section.select_one(".table__title")
                if not section_title:
                    continue
                
                section_name = section_title.text.strip()
                section_ratios = {}
                
                # Buscar filas de ratios
                rows = section.select("tr.table__row")
                for row in rows:
                    name_cell = row.select_one(".table__cell.fixed--column")
                    if not name_cell:
                        continue
                    
                    ratio_name = name_cell.text.strip()
                    values = {}
                    
                    # Obtener valores por año
                    year_cells = row.select(".table__cell:not(.fixed--column)")
                    for i, cell in enumerate(year_cells):
                        # Determinar el año (-5, -4, -3, -2, -1 años desde el actual)
                        year_offset = -5 + i
                        year_value = cell.text.strip()
                        
                        try:
                            # Limpiar el valor
                            if year_value in ['-', 'N/A']:
                                year_value = None
                            else:
                                year_value = year_value.replace(',', '')
                                year_value = float(year_value)
                        except:
                            year_value = None
                        
                        values[str(year_offset)] = year_value
                    
                    section_ratios[ratio_name] = values
                
                ratios[section_name] = section_ratios
            
            logger.info(f"Ratios financieros obtenidos exitosamente para {ticker}")
            return ratios
            
        except Exception as e:
            logger.error(f"Error obteniendo ratios para {ticker}: {str(e)}")
            return {}
    
    def validate_data(self, df: pd.DataFrame, min_columns: int = 3,
                     min_rows: int = 5) -> Tuple[bool, str]:
        """
        Valida la calidad y completitud de los datos extraídos.
        
        Args:
            df: DataFrame a validar
            min_columns: Número mínimo de columnas esperadas
            min_rows: Número mínimo de filas esperadas
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if df.empty:
            return False, "DataFrame está vacío"
        
        # Verificar tamaño mínimo
        if len(df.columns) < min_columns:
            return False, f"El DataFrame tiene menos columnas de lo esperado ({len(df.columns)} < {min_columns})"
        
        if len(df) < min_rows:
            return False, f"El DataFrame tiene menos filas de lo esperado ({len(df)} < {min_rows})"
        
        # Verificar valores nulos excesivos
        null_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if null_percentage > 50:
            return False, f"Demasiados valores nulos ({null_percentage:.2f}%)"
        
        return True, "Datos válidos"
    
    def scrape_company_batch(self, tickers: List[str], output_dir: str) -> Dict[str, Dict]:
        """
        Extrae datos empresariales para múltiples empresas y los guarda.
        
        Args:
            tickers: Lista de símbolos a extraer
            output_dir: Directorio donde guardar los datos
            
        Returns:
            Diccionario con resultados por símbolo
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for ticker in tickers:
            ticker_results = {
                'income_statement': False,
                'balance_sheet': False,
                'ratios': False,
                'sec_filings': False
            }
            
            ticker_dir = os.path.join(output_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            try:
                # 1. Extraer estado de resultados
                income_df = self.scrape_income_statement_investing(ticker)
                valid, message = self.validate_data(income_df)
                
                if valid:
                    income_path = os.path.join(ticker_dir, f"{ticker}_income_statement.csv")
                    income_df.to_csv(income_path, index=False)
                    logger.info(f"Estado de resultados guardado en {income_path}")
                    ticker_results['income_statement'] = True
                else:
                    logger.warning(f"Estado de resultados para {ticker} no válido: {message}")
                
                # 2. Extraer balance general
                self._random_delay()
                balance_df = self.scrape_balance_sheet_investing(ticker)
                valid, message = self.validate_data(balance_df)
                
                if valid:
                    balance_path = os.path.join(ticker_dir, f"{ticker}_balance_sheet.csv")
                    balance_df.to_csv(balance_path, index=False)
                    logger.info(f"Balance general guardado en {balance_path}")
                    ticker_results['balance_sheet'] = True
                else:
                    logger.warning(f"Balance general para {ticker} no válido: {message}")
                
                # 3. Extraer ratios financieros
                self._random_delay()
                ratios = self.scrape_ratios_marketwatch(ticker)
                
                if ratios:
                    ratios_path = os.path.join(ticker_dir, f"{ticker}_ratios.json")
                    with open(ratios_path, 'w') as f:
                        json.dump(ratios, f, indent=2)
                    logger.info(f"Ratios financieros guardados en {ratios_path}")
                    ticker_results['ratios'] = True
                else:
                    logger.warning(f"No se obtuvieron ratios para {ticker}")
                
                # 4. Extraer archivos SEC
                self._random_delay()
                filings = self.scrape_edgar_filings(ticker)
                
                if filings:
                    filings_path = os.path.join(ticker_dir, f"{ticker}_sec_filings.json")
                    with open(filings_path, 'w') as f:
                        json.dump(filings, f, indent=2)
                    logger.info(f"Archivos SEC guardados en {filings_path}")
                    ticker_results['sec_filings'] = True
                else:
                    logger.warning(f"No se obtuvieron archivos SEC para {ticker}")
                
            except Exception as e:
                logger.error(f"Error procesando {ticker}: {str(e)}")
            
            # Guardar resultado para este ticker
            results[ticker] = ticker_results
        
        # Guardar resumen de resultados
        with open(os.path.join(output_dir, "scraping_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results


if __name__ == "__main__":
    # Ejemplo de uso
    scraper = CompanyDataScraper(use_selenium=True, headless=True)
    
    # Lista de ejemplo para pruebas
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    # Extraer y guardar datos
    results = scraper.scrape_company_batch(
        tickers=test_tickers,
        output_dir="data/companies/raw"
    )
    
    print("Resultados de la extracción:")
    for ticker, result in results.items():
        print(f"{ticker}: {result}")