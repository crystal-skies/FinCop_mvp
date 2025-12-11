import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import json
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt # <--- NUEVO: Para gráficos de pastel

# --- CONFIGURACIÓN E INICIO ---
st.set_page_config(page_title="BillMaster AI", page_icon="💰", layout="wide")

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE GEMINI
# -----------------------------------------------------------------------------
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    API_KEY = "tu_api_key" 

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# -----------------------------------------------------------------------------
# 2. FUNCIONES DE IA
# -----------------------------------------------------------------------------
def parse_bill_with_gemini(image):
    prompt = """
    Analiza esta boleta/factura peruana. Devuelve JSON puro:
    {
        "companyName": "Nombre empresa",
        "ruc": "RUC (11 dígitos)",
        "date": "YYYY-MM-DD",
        "total": 0.00,
        "currency": "PEN" o "USD",
        "category": "Categoría (Alimentos, Transporte, Servicios, Ocio, Ropa, Salud, Otros)",
        "items": ["item1", "item2"]
    }
    """
    try:
        response = model.generate_content([prompt, image])
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception as e:
        st.error(f"Error IA: {e}")
        return None

def generate_financial_insights(bills_history):
    if not bills_history: return "Faltan datos."
    history_str = json.dumps(bills_history, indent=2)
    prompt = f"""
    Actúa como 'Billy', una mascota financiera divertida y sabia.
    Analiza este historial (JSON) y dame:
    1. Un patrón de gasto.
    2. Una alerta de gasto hormiga.
    3. Un consejo de ahorro.
    Usa emojis y tono amigable.
    HISTORIAL: {history_str}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Billy está durmiendo (Error de conexión)."

# -----------------------------------------------------------------------------
# 3. BASE DE DATOS LOCAL
# -----------------------------------------------------------------------------
DB_FILE = 'bills_db.json'

def load_bills():
    if not os.path.exists(DB_FILE): return []
    with open(DB_FILE, 'r') as f:
        try: return json.load(f)
        except: return []

def save_bill(bill_data):
    bills = load_bills()
    bill_data['id'] = str(int(time.time() * 1000))
    bill_data['created_at'] = datetime.now().isoformat()
    bills.insert(0, bill_data)
    with open(DB_FILE, 'w') as f: json.dump(bills, f, indent=2)
    return True

def delete_bill(bill_id):
    bills = load_bills()
    bills = [b for b in bills if b.get('id') != bill_id]
    with open(DB_FILE, 'w') as f: json.dump(bills, f, indent=2)

# -----------------------------------------------------------------------------
# 4. COMPONENTE: LA MASCOTA (BILLY)
# -----------------------------------------------------------------------------
def render_mascot_control():
    """
    Muestra un botón flotante (popover) que actúa como mascota.
    Permite cambiar el estado de la sesión 'chart_type'.
    """
    # Inicializar estado si no existe
    if 'chart_type' not in st.session_state:
        st.session_state['chart_type'] = 'Barras'

    # Creamos un contenedor fijo con CSS para simular posición flotante (opcional)
    # Pero para mantener funcionalidad nativa, usaremos st.popover en el dashboard
    
    with st.container():
        # Usamos el popover nativo de Streamlit
        billy = st.popover("🤖 Hablar con Billy", help="Tu asistente financiero")
        
        with billy:
            st.markdown("### ¡Hola! Soy Billy 🤖")
            st.write("¿Te aburren las barras? ¡Cambiemos la vista!")
            
            # Selector de gráficos
            opcion = st.radio(
                "¿Qué gráfico prefieres?",
                ["Barras", "Líneas (Tendencia)", "Pastel (Categorías)"],
                index=0 if st.session_state['chart_type'] == 'Barras' else (1 if st.session_state['chart_type'] == 'Líneas (Tendencia)' else 2)
            )
            
            # Guardamos la elección en el estado
            if opcion != st.session_state['chart_type']:
                st.session_state['chart_type'] = opcion
                st.rerun() # Recargamos para aplicar cambios
            
            st.info("💡 Consejo: El gráfico de pastel es genial para ver en qué categorías se va tu dinero.")

# -----------------------------------------------------------------------------
# 5. VISTA: DASHBOARD
# -----------------------------------------------------------------------------
def view_dashboard():
    # --- CABECERA CON MASCOTA ---
    col_title, col_mascot = st.columns([4, 1])
    with col_title:
        st.title("📊 Mi Panel Financiero")
    with col_mascot:
        # Aquí renderizamos el botón de la mascota
        render_mascot_control()
    
    bills = load_bills()
    
    if not bills:
        st.info("👋 ¡Bienvenido! Aún no tienes datos. Ve a 'Subir Factura' para empezar.")
        return

    df = pd.DataFrame(bills)
    df['total_pen'] = df.apply(lambda x: x['total'] * 3.75 if x['currency'] == 'USD' else x['total'], axis=1)
    df['date_obj'] = pd.to_datetime(df['date'])

    # --- MÉTRICAS ---
    c1, c2, c3, c4 = st.columns(4)
    total_spent = df['total_pen'].sum()
    this_month = df[df['date_obj'].dt.month == datetime.now().month]['total_pen'].sum()
    
    c1.metric("Gasto Total", f"S/ {total_spent:,.2f}")
    c2.metric("Este Mes", f"S/ {this_month:,.2f}")
    c3.metric("Recibos", len(bills))
    c4.metric("Promedio", f"S/ {(total_spent/len(bills)):,.2f}")

    st.divider()

    # --- ZONA DE GRÁFICOS DINÁMICOS (CONTROLADA POR BILLY) ---
    st.subheader(f"Vista de Gastos: {st.session_state.get('chart_type', 'Barras')}")
    
    chart_type = st.session_state.get('chart_type', 'Barras')
    
    # Preparamos datos
    df['mes'] = df['date_obj'].dt.strftime('%Y-%m')
    trend_data = df.groupby('mes')['total_pen'].sum()
    cat_data = df.groupby('category')['total_pen'].sum()

    # Lógica de visualización según la Mascota
    if chart_type == "Barras":
        st.bar_chart(trend_data, color="#6366f1")
        
    elif chart_type == "Líneas (Tendencia)":
        st.line_chart(trend_data, color="#22c55e")
        st.caption("Muestra cómo suben o bajan tus gastos mes a mes.")

    elif chart_type == "Pastel (Categorías)":
        # Streamlit no tiene st.pie_chart nativo, usamos Matplotlib
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.pie(cat_data, labels=cat_data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
        ax.axis('equal') # Para que sea un círculo perfecto
        # Fondo transparente para que se vea bien en Streamlit
        fig.patch.set_alpha(0) 
        st.pyplot(fig)

    st.divider()

    # --- INSIGHTS DE IA ---
    if st.button("✨ Pedir consejo a Billy (IA)"):
        with st.spinner("Billy está pensando... 🐶"):
            advice = generate_financial_insights(bills[:30])
            st.markdown(f"""
            <div style="background-color: #e0e7ff; padding: 20px; border-radius: 15px; border: 2px solid #6366f1; color: #1e1b4b;">
                <strong>🐶 Billy dice:</strong><br><br>
                {advice}
            </div>
            """, unsafe_allow_html=True)

    # --- TABLA DETALLADA ---
    st.subheader("📋 Historial")
    for idx, row in df.iterrows():
        with st.expander(f"{row['date']} - {row['companyName']} (S/ {row['total_pen']:.2f})"):
            c_a, c_b = st.columns(2)
            c_a.write(f"**Categoría:** {row['category']}")
            c_a.write(f"**RUC:** {row.get('ruc', '-')}")
            c_b.write(f"**Total:** {row['currency']} {row['total']}")
            if c_b.button("Eliminar", key=row['id']):
                delete_bill(row['id'])
                st.rerun()

# -----------------------------------------------------------------------------
# 6. VISTA: SUBIR FACTURA
# -----------------------------------------------------------------------------
def view_upload():
    st.title("🧾 Subir Factura")
    c1, c2 = st.columns([1, 1])
    with c1:
        uploaded_file = st.file_uploader("Imagen", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, width=300)
            if "extracted_data" not in st.session_state:
                if st.button("⚡ Analizar", type="primary"):
                    with st.spinner("Procesando..."):
                        data = parse_bill_with_gemini(image)
                        if data:
                            st.session_state['extracted_data'] = data
                            st.rerun()
    with c2:
        if "extracted_data" in st.session_state:
            data = st.session_state['extracted_data']
            st.success("¡Leído!")
            with st.form("confirm"):
                name = st.text_input("Empresa", data.get('companyName',''))
                ruc = st.text_input("RUC", data.get('ruc',''))
                total = st.number_input("Total", float(data.get('total',0.0)))
                cat = st.text_input("Categoría", data.get('category',''))
                curr = st.selectbox("Moneda", ["PEN","USD"], index=0 if data.get('currency')=='PEN' else 1)
                
                # Fecha segura
                try: d_val = pd.to_datetime(data.get('date')).date()
                except: d_val = datetime.now().date()
                date = st.date_input("Fecha", d_val)

                if st.form_submit_button("💾 Guardar"):
                    final = {"companyName": name, "ruc": ruc, "date": str(date), "total": total, "currency": curr, "category": cat, "items": data.get('items', [])}
                    save_bill(final)
                    st.toast("Guardado")
                    del st.session_state['extracted_data']
                    time.sleep(1)
                    st.rerun()
            if st.button("Cancelar"):
                del st.session_state['extracted_data']
                st.rerun()

# -----------------------------------------------------------------------------
# 7. MAIN
# -----------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.title("BillMaster AI")
        menu = st.radio("Ir a:", ["Subir Factura", "Dashboard"])
        st.divider()
        st.caption("Con tecnología Gemini 2.5")

    if menu == "Subir Factura":
        view_upload()
    else:
        view_dashboard()

if __name__ == "__main__":
    main()
