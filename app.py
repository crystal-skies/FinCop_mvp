import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import re
import paddle 
import cv2

# --- CONFIGURACIÓN ---
# Inicializamos Paddle una sola vez. 
# use_gpu=True 
# lang='es' -> Modelo en español
# ocr_version='PP-d' -> El modelo más reciente y preciso
@st.cache_resource
def load_ocr():
    # Configuración explícita (Más segura en Linux)
    paddle.set_device('gpu')
    
    # Volvemos a usar use_gpu=True porque en la v2.7.3 SÍ EXISTE y funciona bien.
    return PaddleOCR(
        use_angle_cls=True, 
        lang='es', 
        use_gpu=True,  # <--- Esto vuelve a funcionar en 2.7.3
        ocr_version='PP-OCRv3' # <--- Usamos el modelo v3 por compatibilidad
    )

ocr = load_ocr()

def clean_text(text):
    return text.strip().upper()

def extract_peru_data(ocr_result):
    """
    Analiza la lista de textos detectados para sacar RUC, Fecha y Total.
    ocr_result es una lista de elementos: [ [[coords], [text, confidence]], ... ]
    """
    
    # Unimos todo el texto para búsquedas generales con Regex
    all_text_lines = [line[1][0] for line in ocr_result]
    full_text_str = " ".join(all_text_lines)
    
    data = {
        "ruc_emisor": None,
        "fecha_emision": None,
        "moneda": "PEN", # Por defecto Soles
        "total": 0.0
    }

    # 1. BUSCAR RUC (Empieza con 10 o 20 y tiene 11 dígitos)
    # Explicación regex: \b inicio de palabra, (10|20) empieza con 10 o 20, \d{9} nueve dígitos más
    ruc_match = re.search(r'\b(10|20)\d{9}\b', full_text_str.replace(" ", "")) 
    if ruc_match:
        data['ruc_emisor'] = ruc_match.group(0)

    # 2. BUSCAR FECHA (Formatos dd/mm/yyyy o dd-mm-yyyy)
    date_match = re.search(r'\d{2}[-/]\d{2}[-/]\d{4}', full_text_str)
    if date_match:
        data['fecha_emision'] = date_match.group(0)

    # 3. BUSCAR TOTAL (Lógica Heurística)
    # Estrategia: Buscar todos los montos con formato de dinero (ej: 120.00, 1,500.50)
    # y usualmente el monto mayor en la factura es el Total.
    
    # Regex para encontrar precios: busca números con punto decimal al final
    amounts = re.findall(r'\d{1,3}(?:,\d{3})*\.\d{2}', full_text_str)
    
    # Limpiamos los montos (quitamos comas) y convertimos a float
    valid_amounts = []
    for amount in amounts:
        try:
            val = float(amount.replace(',', ''))
            valid_amounts.append(val)
        except:
            pass
    
    if valid_amounts:
        # Asumimos que el monto más alto es el total (funciona en el 90% de casos simples)
        data['total'] = max(valid_amounts)
        
    # Detectar si es Dólares
    if "USD" in full_text_str or "$" in full_text_str:
        data['moneda'] = "USD"

    return data, all_text_lines

# --- INTERFAZ WEB (STREAMLIT) ---
st.set_page_config(page_title="OCR Facturas Perú", page_icon="🇵🇪")

st.title("🧾 Extractor de Facturas Perú (GPU)")
st.write("Sube una foto de tu factura o boleta (formato JPG/PNG).")

uploaded_file = st.file_uploader("Seleccionar imagen...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen cargada', use_column_width=True)
    
    if st.button('Extraer Datos'):
        with st.spinner('Procesando con GPU... 🚀'):
            # Paddle espera un array de numpy
            img_array = np.array(image)
            
            # Ejecutar OCR
            result = ocr.ocr(img_array, cls=True)
            
            # Paddle devuelve una lista de resultados por página. Tomamos la primera [0]
            if result[0]:
                extracted_data, raw_lines = extract_peru_data(result[0])
                
                # Mostrar Resultados
                st.success("¡Procesado exitoso!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RUC Emisor", extracted_data['ruc_emisor'] if extracted_data['ruc_emisor'] else "No detectado")
                    st.metric("Fecha", extracted_data['fecha_emision'] if extracted_data['fecha_emision'] else "No detectado")
                with col2:
                    st.metric("Total", f"{extracted_data['total']} {extracted_data['moneda']}")
                
                st.subheader("JSON Resultante")
                st.json(extracted_data)
                
                with st.expander("Ver todo el texto detectado"):
                    st.write(raw_lines)
            else:
                st.warning("No se pudo detectar texto en la imagen.")