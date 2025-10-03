

README
## **1. Descripción del Programa**  
Regression Wizard es una herramienta de ajuste de modelos matemáticos diseñada para analizar datos experimentales mediante regresión no lineal. Permite ajustar múltiples modelos y evaluar su calidad mediante el coeficiente de determinación (R^2).  

### **Características principales:**  
✅ Interfaz gráfica intuitiva (GUI) basada en **Tkinter**.  
✅ Soporta **más de 50 modelos matemáticos** diferentes organizados en categorías.  
✅ Cálculo automático de parámetros óptimos y errores estándar.  
✅ **Análisis de bootstrapping** para estimación de intervalos de confianza.  
✅ Visualización de resultados con **gráficos interactivos** (Matplotlib).  
✅ Exportación de resultados en formatos **TXT y PDF**.  
✅ Evaluación automática de calidad de ajuste (Excelente/Bueno/Moderado/Pobre).  

---

**2. Requisitos del Sistema**  
- **Sistema Operativo:** Windows/macOS/Linux  
- **Python:** 3.8 o superior  
- **Librerías necesarias:**  
  ```bash
  numpy, scipy, matplotlib, tkinter, reportlab
  ```
  Instalación:  
  ```bash
  pip install numpy scipy matplotlib reportlab
  ```

## **3. Especificaciones de los Datos de Entrada**  
### **Formato del archivo de datos:**  
- **Extensión:** `.txt` (archivo de texto plano).  
- **Estructura:** Dos columnas separadas por **espacios/tabulaciones**:  
  ```
  X1    Y1  
  X2    Y2  
  ...   ...  
  ```
- **Requisitos:**  
  - Los valores deben ser **numéricos** (enteros o decimales).  
  - No debe haber encabezados (el programa ignora líneas que comienzan con `#`).  
  - Mínimo de puntos requeridos: **2** (dependiendo del modelo).  

### **Ejemplo de archivo válido:**  
```
# Datos de espectroscopía Raman
100.5  0.25
101.0  0.30
101.5  0.45
102.0  0.80
102.5  1.20
```

---

## **4. Instrucciones de Uso**  
### **Pasos para ejecutar el programa:**  
1. **Ejecutar el script:**  
   ```bash
   python V0.9_fixing_interface.py
   ```
2. **Cargar datos:**  
   - Haga clic en `Cargar Archivo TXT` y seleccione su archivo de datos.  
   - Los datos cargados se mostrarán en el área de texto inferior.  

3. **Seleccionar modelo:**  
   - Elija una **categoría** de la lista desplegable.  
   - Seleccione un **modelo específico** de la segunda lista desplegable.  

4. **Configurar bootstrapping (opcional):**  
   - Active "Usar Bootstrapping" para análisis de incertidumbre.  
   - Configure el número de iteraciones (por defecto: 200).  

5. **Procesar:**  
   - `PROCESAR MODELO SELECCIONADO`: Ajusta el modelo elegido.  
   - `PROCESAR TODOS LOS MODELOS`: Ajusta **todos los modelos** filtrando por R² mínimo.  

6. **Visualizar resultados:**  
   - Los parámetros ajustados y R² aparecerán en la tabla de resultados.  
   - Use los botones para: `Ver Gráfico`, `Ver Estadísticas`, `Ver Residuos`.  

7. **Exportar resultados:**  
   - `Exportar Resultados (TXT)`: Reporte completo en texto.  
   - `Exportar Gráficos (PDF)`: Gráficos y estadísticas en PDF.  

---

## **5. Categorías y Modelos Disponibles**  
| **Categoría**                  | **Número de Modelos** | **Ejemplos** |  
|--------------------------------|----------------------|--------------|  
| **Sigmoidales**                | 13 modelos           | Logística, Gompertz, Hill, Chapman, Weibull |  
| **Polinomiales**               | 3 modelos            | Lineal, Cuadrático, Cúbico |  
| **Inversos polinomiales**      | 3 modelos            | Inverso 1º-3º orden |  
| **Modelos de Campana**         | 12 modelos           | Gaussiana, Lorentzian, Pseudo-Voigt, Log-Normal |  
| **Decaimiento Exponencial**    | 8 modelos            | Simple, Doble, Triple exponencial |  
| **Crecimiento exponencial limitado** | 6 modelos       | Crecimiento simple y doble |  
| **Modelos de Potencia**        | 8 modelos            | Potencia simple, simétrica, Pareto |  
| **Ciencia-Bio**                | 23 modelos           | Michaelis-Menten, Dose-Response, Crecimiento poblacional |  
| **Ingeniería 2D**              | 32 modelos           | Dispersión óptica, Steinhart-Hart, modelos de motor |  
| **Modelos NIST**               | 14 modelos           | Hyperbolic Tangent, Burr Model, Kohout Model |  

---

## **6. Interpretación de Resultados**  
- **R²:** Valor entre 0 y 1. Mientras más cercano a 1, mejor el ajuste.  
  - Excelente: R² ≥ 0.9 (verde)  
  - Bueno: 0.7 ≤ R² < 0.9 (amarillo)  
  - Moderado: 0.5 ≤ R² < 0.7 (naranja)  
  - Pobre: R² < 0.5 (rojo)  

- **Bootstrapping:** Proporciona intervalos de confianza robustos para los parámetros.  

- **Evaluación de residuos:**  
  - Gráficos de residuos vs variable independiente.  
  - Residuos normalizados para detectar valores atípicos.  

---

## **7. Características Avanzadas**  
### **Bootstrapping:**  
- Estimación no paramétrica de intervalos de confianza.  
- Visualización de distribuciones de parámetros.  
- Múltiples gráficos de análisis de incertidumbre.  

### **Exportación PDF:**  
- Portada con información general.  
- Gráficos de ajuste para cada modelo.  
- Tablas de parámetros y estadísticas.  
- Análisis de residuos y distribuciones bootstrap.  

### **Procesamiento por lotes:**  
- Filtrado por R² mínimo.  
- Procesamiento automático de múltiples modelos.  
- Organización de resultados por categoría.  

---

## **8. Ejemplo de Salida Exportada (TXT)**  
```plaintext
RESULTADOS DE REGRESIÓN - REGRESSION WIZARD
===========================================
Fecha: 2024-05-16
Total modelos procesados: 15

CATEGORÍA: MODELOS DE CAMPANA
--------------------------------------------------
Modelo: Gaussiana 4 parámetros
R²: 0.985 (Excelente)
Parámetros: a=1.25 ± 0.03, b=0.5 ± 0.01, x0=102.0 ± 0.1, y0=0.1 ± 0.02
IC 95% bootstrap: [1.22, 1.28], [0.48, 0.52], [101.9, 102.1], [0.08, 0.12]
--------------------------------------------------
```

---

## **9. Notas Adicionales**  
- **Precisión:** Los resultados dependen de la calidad y cantidad de datos.  
- **Requisitos mínimos:** Varían según el modelo (2-11 puntos).  
- **Validación biológica:** Los modelos de ciencia-bio incluyen validación de parámetros biológicamente significativos.  
- **Rendimiento:** El bootstrapping puede aumentar el tiempo de procesamiento.  

**Versión:** 0.9 - Regression Wizard  
**Última actualización:** 2024
```

Key updates made:
1. **Version number**: Updated to v0.9
2. **Model count**: Reflected the expanded model library (50+ models)
3. **New categories**: Added all the new categories from the code
4. **Bootstrapping feature**: Added comprehensive documentation
5. **PDF export**: Documented the new PDF export capability
6. **Advanced features**: Added sections for bootstrapping, PDF export, and batch processing
7. **Interface updates**: Reflected the current GUI organization with tabs and buttons
8. **Evaluation scale**: Updated to include the 4-tier evaluation system
9. **Requirements**: Added reportlab to the required libraries

The README now accurately reflects the capabilities and features of your v0.9 Regression Wizard program.
