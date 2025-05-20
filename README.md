## **1. Descripción del Programa**  
Regression Wizard es una herramienta de ajuste de modelos matemáticos diseñada para analizar datos experimentales mediante regresión no lineal. Permite ajustar múltiples modelos como sigmoidales, polinomiales, inversos y modelos de campana, Crecimiento exponencial limitada y Modelos de potencia y evaluar su calidad mediante el coeficiente de determinación (R^2).  

### **Características principales:**  
✅ Interfaz gráfica intuitiva (GUI) basada en **Tkinter**.  
✅ Soporta **31 modelos matemáticos** diferentes.  
✅ Cálculo automático de parámetros óptimos y errores estándar.  
✅ Visualización de resultados con **gráficos interactivos** (Matplotlib).  
✅ Exportación de resultados en formato **TXT**.  

---

## **2. Requisitos del Sistema**  
- **Sistema Operativo:** Windows/macOS/Linux  
- **Python:** 3.8 o superior  
- **Librerías necesarias:**  
  ```bash
  numpy, scipy, matplotlib, tkinter
  ```
  Instalación:  
  ```bash
  pip install numpy scipy matplotlib
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
   python nombre_del_archivo.py
   ```
2. **Cargar datos:**  
   - Haga clic en `Cargar Archivo TXT` y seleccione su archivo de datos.  
   - Los datos cargados se mostrarán en el área de texto inferior.  

3. **Seleccionar modelo:**  
   - Elija una **categoría** (ej: "Modelos de Campana").  
   - Seleccione un **modelo específico** (ej: "Gaussiana 4 parámetros").  

4. **Procesar:**  
   - `PROCESAR MODELO SELECCIONADO`: Ajusta el modelo elegido.  
   - `PROCESAR TODOS LOS MODELOS`: Ajusta **todos los modelos de la categoría actual**.  

5. **Visualizar resultados:**  
   - Los parámetros ajustados y \( R^2 \) aparecerán en la tabla.  
   - Haga clic en `Ver Gráfico` para visualizar el ajuste.  

6. **Exportar resultados:**  
   - Use `Exportar Resultados (TXT)` para guardar un reporte.  

---

## **5. Categorías y Modelos Disponibles**  
| **Categoría**           | **Modelos Incluidos**                               | **Mín. Puntos** |  
|--------------------------|----------------------------------------------------|----------------|  
| **Sigmoidales**          | Logística, Gompertz, Hill, Chapman (3-4-5 params)  | 2-5            |  
| **Polinomiales**         | Lineal, Cuadrático, Cúbico                         | 2-4            |  
| **Inversos polinomiales**| Inverso 1º-3º orden                                | 2-4            |  
| **Modelos de Campana**   | Gaussiana, Lorentzian, Pseudo-Voigt, Log-Normal    | 3-5            |  

---

## **6. Interpretación de Resultados**  
- **\( R^2 \):** Valor entre 0 y 1. Mientras más cercano a 1, mejor el ajuste.  
  - Excelente: \( R^2 \geq 0.9 \) (verde)  
  - Bueno: \( 0.7 \leq R^2 < 0.9 \) (amarillo)  
  - Pobre: \( R^2 < 0.5 \) (rojo)  
- **Parámetros:** Se muestran con su **error estándar** (ej: `a = 1.25 ± 0.03`).  

---

## **7. Ejemplo de Salida Exportada (TXT)**  
```plaintext
RESULTADOS DE REGRESIÓN - REGRESSION WIZARD
===========================================
Fecha: 2024-05-16
Total modelos procesados: 5

CATEGORÍA: MODELOS DE CAMPANA
--------------------------------------------------
Modelo: Gaussiana 4 parámetros
R²: 0.985 (Excelente)
Parámetros: a=1.25 ± 0.03, b=0.5 ± 0.01, x0=102.0 ± 0.1, y0=0.1 ± 0.02
--------------------------------------------------
```

---

## **8. Notas Adicionales**  
- **Precisión:** Los resultados dependen de la calidad y cantidad de datos.  
- **Advertencias:**  
  - Los modelos inversos no aceptan valores `x = 0`.  
  - Para modelos complejos (ej: Pseudo-Voigt), se requieren más puntos.  

---
