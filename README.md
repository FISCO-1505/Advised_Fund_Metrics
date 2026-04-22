# 📈 Advised Funds Metrics & Portfolio Analytics

### **Quantitative Financial for Multi-Asset Analysis**

Esta aplicación es una solución avanzada de **Business Intelligence Financiero** desarrollada en Python y Streamlit. Automatizando el cálculo de métricas de riesgo/retorno y la generación de reportes institucionales.

---

## 🚀 Características Principales

### 1. **Análisis Multi-Tópico**
* **Funds & Commodities:** Análisis individual de activos con comparación contra Benchmarks dinámicos.
* **Portfolio Analytics:** Motor de cálculo dinámico para carteras basadas en unidades nominales y jerarquías (Standard y Combo).
* **Returns Tables:** Generación de reportes de rendimiento YTD con integración de gráficos de alta calidad.

### 2. **Motor de Métricas Cuantitativas**
La app integra un kit de herramientas matemáticas para calcular:
* **Riesgo:** Volatilidad Anualizada, Downside Deviation, VaR Histórico (99%), Max Drawdown y Tracking Error.
* **Desempeño:** Sharpe Ratio, Sortino Ratio, Treynor Ratio e Information Ratio.
* **Atribución:** Beta, Correlación, R², Alpha de Jensen y Alpha Activo.

### 3. **Arquitectura de Datos Robusta**
* **Validación de Estructura:** Verifica la integridad de archivos Excel (hojas, columnas, tipos de datos,etc).
* **Gestión de Fechas:** Soporte nativo para periodos MTD, YTD, 1Y, Since Inception y rangos personalizados.
* **Caché Avanzada:** Uso intensivo de `@st.cache_data` y `@st.cache_resource` para garantizar una respuesta inmediata incluso con grandes volúmenes de datos.

---

## 🛠 Estructura del Proyecto (Lógica de Módulos)

El sistema se divide en cuatro capas lógicas fundamentales:

### **A. Capa de Interfaz (Motor Principal)**
Controla el flujo de usuario y la navegación.
* `main()`: Punto de entrada con autenticación segura y gestión de sesión.
* `contenido_principal()`: Gestiona el Sidebar, la carga de archivos y el ruteo entre secciones.



### **B. Capa de Procesamiento (`kit_f_principales`)**
Orquesta los cálculos pesados.
* `Funds_Commodity()` / `Portfolio()`: Funciones que transforman precios en indicadores financieros.
* `procesar_analisis()`: Unifica la lógica de UI para el filtrado de activos y estadísticas.

### **C. Capa de Métricas (`kit_metricas`)**
Contiene la lógica matemática pura.
* `df_returns()`: Cálculo de variaciones porcentuales respetando NaNs originales.
* `rolling_vol()` & `sharpe_ratio()`: Algoritmos de riesgo expansivo.
* `portfolio_Prices()`: Reconstrucción de precios históricos basada en nominales variables.

### **D. Capa de Reporting (`kit_f_secundarias`)**
Maneja las salidas y formatos.
* **Visualización:** Generación de gráficos suavizados (Spline Interpolation) con Matplotlib.
* **Exportación:** Generación de archivos Excel institucionales con formatos personalizados.

---

## 📊 Flujo de Usuario (UX/UI)

1.  **Autenticación:** El usuario ingresa mediante una clave segura gestionada por `FISCO_Sources`.
2.  **Carga y Validación:** Se sube un archivo Excel. El sistema valida automáticamente que la estructura sea correcta antes de permitir el acceso.
3.  **Configuración de Análisis:** * Selección de Tópico (Funds/Portfolio).
    * Selección de Activos (filtros por Ticker o nombres amigables).
    * Selección de Métricas (KPIs específicos).
    * Definición de Periodo (MTD, YTD, etc.).
4.  **Ejecución:** Al presionar "Load Process", se despliega una tabla interactiva con los resultados formateados.
5.  **Descarga:** Generación de reportes PDF/Excel con un solo clic.

---

## 📋 Requisitos del Archivo de Entrada (Excel)

Para un correcto funcionamiento, el archivo debe contener las siguientes hojas mínimas:
* `Info`: Metadatos de activos (Ticker, Type, Benchmark asociado).
* `Prices`: Precios históricos diarios.
* `Nominals`: Histórico de posiciones para la reconstrucción de portafolios.
* `Weights`: Ponderaciones para el cálculo de Benchmarks.

---

## 💻 Tecnologías Utilizadas

* **Core:** Python 3.x
* **Frontend:** Streamlit
* **Data Analysis:** Pandas, NumPy, Scipy
* **Visualización:** Matplotlib (con suavizado Spline)
* **Excel Engine:** XlsxWriter / Openpyxl

---

## 🔒 Seguridad y Rendimiento

* **Timeout:** La sesión se cierra automáticamente tras 3600 segundos de inactividad.
* **Optimización:** El uso de memoria se optimiza filtrando los DataFrames antes de realizar cálculos expansivos, asegurando estabilidad en despliegues de nube.

---

> **Nota:** Este software es una herramienta de análisis cuantitativo. Los resultados generados son para fines informativos y de asesoría, basados en los datos proporcionados en el archivo de entrada.