import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import re
import paddle 
import cv2
from difflib import SequenceMatcher # <--- IMPORTANTE: Para comparar textos parecidos

# --- CONFIGURACIÓN ---
@st.cache_resource
def load_ocr():
    paddle.set_device('gpu')
    return PaddleOCR(
        use_angle_cls=True, 
        lang='es', 
        use_gpu=True, 
        ocr_version='PP-OCRv3', 
        show_log=False,
        det_db_thresh=0.1,  
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=2.0
    )

ocr = load_ocr()

# --- VALIDACIÓN RUC (SUNAT) ---
def validar_ruc(ruc_str):
    if not ruc_str or len(ruc_str) != 11: return False
    try:
        factores = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
        suma = sum([int(ruc_str[i]) * factores[i] for i in range(10)])
        residuo = suma % 11
        d_verificacion = 11 - residuo
        if d_verificacion == 10: d_verificacion = 0
        elif d_verificacion == 11: d_verificacion = 1
        return d_verificacion == int(ruc_str[10])
    except: return False

# --- PREPROCESAMIENTO ---
def preprocess_image(image_pil):
    img = np.array(image_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(gray)
    return enhanced_img

def extract_money(text):
    match = re.search(r'\d{1,3}(?:,\d{3})*\.\d{2}', text)
    if match:
        try: return float(match.group(0).replace(',', ''))
        except: return None
    return None

# --- DICCIONARIO ---
DB_EMPRESAS = {
    "DOLLARCITY": "20606109343",
    "TIENDAS PERUANAS": "20493020618", # Oechsle
    "SUPERMERCADOS PERUANOS": "20100070970", # Plaza Vea / Vivanda
    "TOTTUS": "20508565934",
    "SODIMAC": "20503870123",
    "CINEPLANET": "20429683581",
    "REPSOL": "20503692195",
    "TAMBO": "20563032890",
    "RIPLEY": "20337564373",
    "FALABELLA": "20100128056",
}

def similar(a, b):
    """Calcula qué tan parecidos son dos textos (0.0 a 1.0)"""
    return SequenceMatcher(None, a, b).ratio()

def extract_peru_data(ocr_result, full_text_combined):
    data = {"ruc_emisor": None, "fecha_emision": None, "moneda": "PEN", "total": 0.0}
    
    # Obtenemos las primeras 10 líneas para buscar el nombre de la empresa
    header_lines = [line[1][0].upper() for line in ocr_result[:10]]
    
    # =========================================================================
    # ESTRATEGIA 1: BÚSQUEDA BORROSA EN DICCIONARIO (NIVEL DIOS)
    # =========================================================================
    found_company_ruc = None
    
    for line in header_lines:
        for empresa_db, ruc_db in DB_EMPRESAS.items():
            # Si la similitud es mayor al 60% (0.6), asumimos que es esa empresa
            # Ejemplo: "ColTcity" vs "DOLLARCITY" suele dar alto
            if similar(line, empresa_db) > 0.6 or empresa_db in line:
                found_company_ruc = ruc_db
                print(f"✅ Empresa detectada: {empresa_db} (Leído: {line}) -> RUC: {ruc_db}")
                break
        if found_company_ruc: break
    
    if found_company_ruc:
        data['ruc_emisor'] = found_company_ruc
    
    # =========================================================================
    # ESTRATEGIA 2: BÚSQUEDA CONTEXTUAL (NIVEL ORO)
    # =========================================================================
    # Si NO encontramos empresa en DB, buscamos el RUC en el papel
    if not data['ruc_emisor']:
        
        # Paso A: Buscar explícitamente "RUC" + NÚMERO
        # Esta regex caza "RUC20..." (pegado) o "R.U.C.: 20..." (separado)
        match_con_etiqueta = re.search(r'RUC.*?(\d{11})', full_text_combined.upper().replace(" ", ""))
        
        if match_con_etiqueta:
            candidato = match_con_etiqueta.group(1)
            # Solo si empieza con 10 o 20 (emisores válidos comunes) y valida SUNAT
            if candidato.startswith(('10', '20', '15', '17')) and validar_ruc(candidato):
                 data['ruc_emisor'] = candidato
        
        # Paso B: Si falló A, buscar cualquier número válido (NIVEL BRONCE)
        # Pero cuidado con tu error del 1085... Priorizamos los que empiezan con 20 (Empresas)
        if not data['ruc_emisor']:
            todos_los_rucs = re.findall(r'(?<!\d)(10|15|16|17|20)\d{9}(?!\d)', full_text_combined)
            
            # Preferencia a RUCs que empiezan con 20 (Empresas) sobre 10 (Personas)
            rucs_20 = [r for r in todos_los_rucs if r.startswith('20') and validar_ruc(r)]
            rucs_10 = [r for r in todos_los_rucs if r.startswith('10') and validar_ruc(r)]
            
            if rucs_20:
                data['ruc_emisor'] = rucs_20[0]
            elif rucs_10:
                data['ruc_emisor'] = rucs_10[0]

    # --- FECHA ---
    date_regex = r'(\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})'
    date_match = re.search(date_regex, full_text_combined)
    if date_match:
        data['fecha_emision'] = date_match.group(0)

    # --- MONEDA ---
    if "SOLES" in full_text_combined.upper(): data['moneda'] = "PEN"
    elif "USD" in full_text_combined.upper(): data['moneda'] = "USD"

    # --- TOTAL ---
    found_total = False
    for i, line in enumerate(ocr_result):
        text_origin = line[1][0].upper()
        if "TOTAL" in text_origin and "SUB" not in text_origin:
            amount = extract_money(text_origin)
            if amount:
                data['total'] = amount; found_total = True; break 
            if i + 1 < len(ocr_result):
                amount_next = extract_money(ocr_result[i+1][1][0])
                if amount_next: data['total'] = amount_next; found_total = True; break
    
    if not found_total:
        for i, line in enumerate(ocr_result):
            text_origin = line[1][0].upper()
            if "IMPORTE" in text_origin or "A PAGAR" in text_origin:
                amount = extract_money(text_origin)
                if amount: data['total'] = amount; break
                if i + 1 < len(ocr_result):
                    amount_next = extract_money(ocr_result[i+1][1][0])
                    if amount_next: data['total'] = amount_next; break
    
    raw_lines = [line[1][0] for line in ocr_result]
    return data, raw_lines

# --- INTERFAZ WEB ---
st.set_page_config(page_title="OCR Pro", page_icon="🇵🇪")
st.title("🧾 Extractor Inteligente (Fuzzy + Contexto)")

uploaded_file = st.file_uploader("Subir imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    col1.image(image, caption='Original', use_container_width=True)
    
    if st.button('Extraer Datos'):
        with st.spinner('Procesando...'):
            processed_img = preprocess_image(image)
            col2.image(processed_img, caption='Procesada', use_container_width=True)
            
            result = ocr.ocr(processed_img, cls=True)
            
            if result[0]:
                raw_lines = [line[1][0] for line in result[0]]
                full_text = " ".join(raw_lines)
                
                extracted_data, lines_returned = extract_peru_data(result[0], full_text)
                
                st.success("¡Lectura completada!")
                c1, c2, c3 = st.columns(3)
                c1.metric("RUC", extracted_data['ruc_emisor'] if extracted_data['ruc_emisor'] else "No detectado")
                c2.metric("Fecha", extracted_data['fecha_emision'])
                c3.metric("Total", f"{extracted_data['moneda']} {extracted_data['total']}")
                
                st.json(extracted_data)
                with st.expander("Ver texto crudo"):
                    st.write(lines_returned)
            else:
                st.warning("No se detectó texto.")