# Sistema ETL de Análisis Financiero Multi-Mercado

## Descripción
Pipeline ETL automatizado que recopila, procesa y analiza datos financieros de múltiples fuentes (mercado bursátil, análisis empresarial y criptomonedas). El sistema integra web scraping y APIs públicas, procesa los datos con Apache Airflow y genera visualizaciones avanzadas en Power BI para identificar tendencias, correlaciones y oportunidades de inversión.

## Características
- Extracción de datos de múltiples fuentes financieras
- Procesamiento y análisis de datos con indicadores técnicos
- Detección de correlaciones entre diferentes tipos de activos
- Análisis de tendencias y ciclos de mercado
- Carga en base de datos estructurada
- Orquestación completa con Apache Airflow

## Requisitos
- Python 3.9+
- Apache Airflow
- Claves de API para servicios financieros
- Ver requirements.txt para dependencias completas

## Instalación
1. Clonar este repositorio
2. Crear entorno virtual: \python -m venv venv\
3. Activar entorno: \source venv/bin/activate\ (Linux/Mac) o \env\\Scripts\\activate\ (Windows)
4. Instalar dependencias: \pip install -r requirements.txt\
5. Copiar \.env.example\ a \.env\ y configurar API keys
6. Inicializar base de datos: \python -m src.database.create_database_schema\

## Uso
- Extracción manual: \python -m src.extractors.stock_extractor\
- Ejecutar mediante Airflow: Activar DAG \master_etl_pipeline\

## Estructura
El proyecto sigue una estructura modular con componentes independientes para extracción, transformación y carga.
