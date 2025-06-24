import streamlit as st
import openai
import fitz  # PyMuPDF
import tiktoken
import numpy as np
import faiss
import pickle
import json
import re
import os
from typing import List, Tuple, Optional
import time

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Nutrición Embarazo - LLM + RAG",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🤰 Chatbot de Nutrición para Embarazadas")
st.markdown("### Sistema LLM + RAG para consultas sobre alimentación durante el embarazo")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # API Key de OpenAI
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Ingresa tu API key de OpenAI"
    )
    
    if not api_key:
        st.warning("⚠️ Por favor ingresa tu API key de OpenAI para continuar")
        st.stop()
    
    # Configuración del modelo
    st.subheader("🔧 Parámetros del Modelo")
    model_name = st.selectbox(
        "Modelo LLM",
        ["gpt-4o-mini", "gpt-3.5-turbo"],
        index=0
    )
    
    top_k = st.slider(
        "Número de chunks a recuperar (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Cantidad de fragmentos de texto más relevantes a usar como contexto"
    )
    
    use_rag = st.checkbox(
        "Usar RAG",
        value=True,
        help="Activar/desactivar el sistema de recuperación aumentada de información"
    )
    
    show_chunks = st.checkbox(
        "Mostrar chunks recuperados",
        value=False,
        help="Mostrar los fragmentos de texto utilizados para generar la respuesta"
    )

# Inicializar cliente OpenAI
@st.cache_resource
def init_openai_client(api_key: str):
    """Inicializa el cliente de OpenAI"""
    return openai.OpenAI(api_key=api_key)

# Función para extraer texto del PDF
def extract_text_from_pdf(
    pdf_path: str,
    use_blocks: bool = True,
    min_length: int = 40,
    exclude_headers_footers: bool = True,
    header_height: int = 60,
    footer_height: int = 60,
    skip_pages: Optional[List[int]] = None,
    start_at_page: int = 0,
    stop_at_page: Optional[int] = None,
    omit_texts: Optional[List[str]] = None
) -> List[str]:
    """
    Extrae texto de un PDF usando PyMuPDF, ordenando bloques visuales y controlando
    rango de páginas y omisión de textos específicos.
    """
    doc = fitz.open(pdf_path)
    all_chunks = []
    skip_pages = skip_pages or []
    omit_texts = [t.strip().lower() for t in (omit_texts or [])]

    start_marker = "lunes martes miércoles jueves viernes sábado domingo"
    end_marker = "oriente sobre cómo poner en práctica la recomendación, mencionando que:"
    skipping = False

    page_range = range(start_at_page, stop_at_page if stop_at_page is not None else len(doc))

    for i in page_range:
        if i in skip_pages:
            continue

        page = doc[i]
        page_height = page.rect.height

        if use_blocks:
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            for b in blocks:
                x0, y0, x1, y1, text = b[:5]
                if not text or len(text.strip()) < min_length:
                    continue
                if exclude_headers_footers and (y0 < header_height or y1 > page_height - footer_height):
                    continue
                clean_text = re.sub(r'\s+', ' ', text.strip().replace("\n", " ").replace("\t", " "))
                lower_clean = clean_text.lower()
                if lower_clean in omit_texts:
                    continue
                if start_marker in lower_clean:
                    skipping = True
                    continue
                if end_marker in lower_clean:
                    skipping = False
                    continue
                if skipping:
                    continue
                all_chunks.append(clean_text)
        else:
            text = page.get_text()
            if text and len(text.strip()) >= min_length:
                clean_text = re.sub(r'\s+', ' ', text.strip().replace("\n", " ").replace("\t", " "))
                lower_clean = clean_text.lower()
                if lower_clean in omit_texts:
                    continue
                if start_marker in lower_clean:
                    skipping = True
                    continue
                if end_marker in lower_clean:
                    skipping = False
                    continue
                if skipping:
                    continue
                all_chunks.append(clean_text)

    return all_chunks

# Función para generar embeddings
def embed_texts_openai(texts: List[str], client) -> np.ndarray:
    """Genera embeddings usando OpenAI"""
    embeddings = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(texts):
        status_text.text(f"Generando embedding {i+1}/{len(texts)}...")
        try:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error generando embedding: {e}")
            return np.array([])
        
        progress_bar.progress((i + 1) / len(texts))
    
    progress_bar.empty()
    status_text.empty()
    return np.array(embeddings).astype("float32")

# Función para crear índice FAISS
def create_faiss_index(embeddings: np.ndarray):
    """Crea un índice FAISS para búsqueda rápida"""
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Función para guardar embeddings
def save_embeddings_to_pkl(vectors: np.ndarray, chunks: List[str], file_path: str = "embeddings_data.pkl"):
    """Guarda los embeddings y chunks en un archivo pickle"""
    data = {
        "vectors": np.array(vectors).astype("float64"),
        "chunks": chunks
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    st.success(f"✅ Embeddings guardados en {file_path}")

# Función para cargar embeddings
def load_embeddings_from_pkl(file_path: str = "embeddings_data.pkl") -> Tuple[np.ndarray, List[str]]:
    """Carga los embeddings y chunks desde un archivo pickle"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    st.success(f"📂 Cargados {len(data['chunks'])} embeddings desde {file_path}")
    return data["vectors"], data["chunks"]

# Función para resumir historial
def summarize_history(history: List[dict], client, max_turns: int = 2) -> str:
    """Resume la conversación contenida en 'history'"""
    if len(history) < max_turns * 2:
        return "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)

    history_text = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
    resumen_prompt = (
        f"Resume la siguiente conversación entre usuario y chatbot sobre alimentación durante el embarazo, "
        f"conservando los puntos clave y detalles importantes:\n\n{history_text}\n\nResumen:"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en resumir conversaciones de nutrición para embarazadas."},
                {"role": "user", "content": resumen_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generando resumen: {e}")
        return history_text

# Función principal para recuperar y responder
def retrieve_and_answer_openai(
    question: str,
    chunks: List[str],
    index,
    chunk_embeddings: np.ndarray,
    client,
    summary: Optional[str] = None,
    top_k: int = 5,
    use_rag: bool = True,
    print_chunks: bool = False,
    model_name: str = "gpt-4o-mini"
) -> str:
    """Función principal para recuperar información y generar respuesta"""
    
    if use_rag:
        # Generar embedding de la pregunta
        try:
            q_embedding = client.embeddings.create(
                input=question,
                model="text-embedding-3-small"
            ).data[0].embedding
            q_vector = np.array(q_embedding).astype("float32").reshape(1, -1)
            
            # Buscar chunks más similares
            D, I = index.search(q_vector, top_k)
            
            if print_chunks:
                st.subheader("📄 Chunks recuperados:")
                for rank, idx in enumerate(I[0]):
                    with st.expander(f"Chunk #{idx} (Distancia: {D[0][rank]:.4f})"):
                        st.write(chunks[idx])
            
            context = "\n---\n".join([chunks[i] for i in I[0]])

            if summary:
                prompt = f"""Resumen de la conversación previa:
{summary}

Usa la siguiente información para responder la pregunta de forma clara y completa. Solo responde con base en la información proporcionada, no busques datos en otras fuentes.
Información:
{context}

Pregunta: {question}
Respuesta:"""
            else:
                prompt = f"""Usa la siguiente información para responder la pregunta de forma clara y completa. Solo responde con base en la información proporcionada, no busques datos en otras fuentes.
Información:
{context}

Pregunta: {question}
Respuesta:"""
        except Exception as e:
            st.error(f"Error en la recuperación RAG: {e}")
            return "Lo siento, hubo un error al procesar tu pregunta."
    else:
        if summary:
            prompt = f"""Resumen de la conversación previa:
{summary}

Responde de forma clara y completa la siguiente pregunta sobre alimentación durante el embarazo. Si no tienes información suficiente, dilo claramente.

Pregunta: {question}
Respuesta:"""
        else:
            prompt = f"""Responde de forma clara y completa la siguiente pregunta sobre alimentación durante el embarazo. Si no tienes información suficiente, dilo claramente.

Pregunta: {question}
Respuesta:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un chatbot para resolver dudas de la alimentación durante el embarazo "
                        "que responde en base a la información proporcionada de contexto. "
                        "Si la información no es suficiente, entonces dices que no puedes responder esa pregunta. "
                        "Si algún fragmento de la información proporcionada no está relacionada con la pregunta, ignóralo al hacer la respuesta. "
                        "Responde siempre en español de manera clara y amigable."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generando respuesta: {e}")
        return "Lo siento, hubo un error al generar la respuesta."

# Inicializar el cliente
client = init_openai_client(api_key)

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Procesamiento", "ℹ️ Información"])

with tab1:
    st.header("💬 Chat con el Asistente de Nutrición")
    
    # Inicializar historial en session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summary" not in st.session_state:
        st.session_state.summary = None
    
    # Cargar embeddings y crear índice
    @st.cache_resource
    def load_system_data():
        """Carga los datos del sistema (embeddings, chunks, índice)"""
        try:
            # Intentar cargar embeddings existentes
            if os.path.exists("embeddings_guia_alimentaria_v2.pkl"):
                embeddings, chunks = load_embeddings_from_pkl("embeddings_guia_alimentaria_v2.pkl")
                index = create_faiss_index(embeddings)
                return embeddings, chunks, index
            else:
                st.warning("No se encontraron embeddings pre-procesados. Por favor, procesa el PDF primero en la pestaña 'Procesamiento'.")
                return None, None, None
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return None, None, None
    
    embeddings, chunks, faiss_index = load_system_data()
    
    if embeddings is not None and chunks is not None and faiss_index is not None:
        # Mostrar historial de chat
        st.subheader("📝 Historial de Conversación")
        
        # Contenedor para el historial
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
        
        # Input para nueva pregunta
        st.subheader("❓ Haz tu pregunta")
        
        # Ejemplos de preguntas
        with st.expander("💡 Ejemplos de preguntas"):
            st.write("""
            - ¿Puedo comer frutos rojos durante el embarazo?
            - ¿Qué alimentos debo evitar durante el embarazo?
            - ¿Cuánta agua debo consumir diariamente?
            - ¿Puedo comer pescado durante el embarazo?
            - ¿Qué frutas son recomendadas durante el embarazo?
            """)
        
        # Formulario para pregunta
        with st.form("chat_form"):
            user_question = st.text_area(
                "Tu pregunta:",
                placeholder="Escribe tu pregunta sobre nutrición durante el embarazo...",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button("🚀 Enviar", use_container_width=True)
            with col2:
                clear_button = st.form_submit_button("🗑️ Limpiar Chat", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.summary = None
            st.rerun()
        
        if submit_button and user_question.strip():
            # Agregar pregunta al historial
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Mostrar pregunta del usuario
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner("🤔 Pensando..."):
                    response = retrieve_and_answer_openai(
                        question=user_question,
                        chunks=chunks,
                        index=faiss_index,
                        chunk_embeddings=embeddings,
                        client=client,
                        summary=st.session_state.summary,
                        top_k=top_k,
                        use_rag=use_rag,
                        print_chunks=show_chunks,
                        model_name=model_name
                    )
                
                st.write(response)
            
            # Agregar respuesta al historial
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Actualizar resumen si el historial es largo
            if len(st.session_state.chat_history) > 6:  # Más de 3 turnos
                with st.spinner("📝 Actualizando resumen de conversación..."):
                    st.session_state.summary = summarize_history(st.session_state.chat_history, client)
            
            st.rerun()
    
    else:
        st.info("👆 Ve a la pestaña 'Procesamiento' para cargar y procesar el PDF de la guía alimentaria.")

with tab2:
    st.header("📊 Procesamiento de Datos")
    
    st.subheader("📄 Extracción de Texto del PDF")
    
    # Upload del PDF
    uploaded_file = st.file_uploader(
        "Sube el archivo PDF de la guía alimentaria",
        type=['pdf'],
        help="Sube el archivo PDF que contiene la información nutricional"
    )
    
    if uploaded_file is not None:
        # Guardar archivo temporalmente
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("✅ PDF cargado correctamente")
        
        # Configuración de extracción
        st.subheader("⚙️ Configuración de Extracción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_blocks = st.checkbox("Usar bloques visuales", value=True)
            min_length = st.number_input("Longitud mínima del chunk", min_value=1, value=40)
            exclude_headers_footers = st.checkbox("Excluir encabezados y pies de página", value=True)
        
        with col2:
            start_page = st.number_input("Página inicial", min_value=0, value=6)
            end_page = st.number_input("Página final", min_value=0, value=78)
            header_height = st.number_input("Altura del encabezado (pts)", min_value=0, value=0)
            footer_height = st.number_input("Altura del pie de página (pts)", min_value=0, value=50)
        
        # Textos a omitir
        omit_texts = st.multiselect(
            "Textos a omitir:",
            options=[
                "¿CÓMO ESTAMOS EN MÉXICO?",
                "¿POR QUÉ ES IMPORTANTE ESTA RECOMENDACIÓN?",
                "En la salud:",
                "En las mujeres:",
                "En las y los bebés:",
                "En la cultura y en la sociedad:",
                "En el ambiente:",
                "En la economía, en la cultura y en la sociedad:",
                "En el o la bebé gestante",
                "En mujeres embarazadas:",
                "En mujeres en periodo de lactancia y posparto:",
                "En las y los recién nacidos:"
            ],
            default=[
                "¿CÓMO ESTAMOS EN MÉXICO?",
                "¿POR QUÉ ES IMPORTANTE ESTA RECOMENDACIÓN?",
                "En la salud:",
                "En las mujeres:",
                "En las y los bebés:",
                "En la cultura y en la sociedad:",
                "En el ambiente:",
                "En la economía, en la cultura y en la sociedad:",
                "En el o la bebé gestante",
                "En mujeres embarazadas:",
                "En mujeres en periodo de lactancia y posparto:",
                "En las y los recién nacidos:"
            ]
        )
        
        # Botón para procesar
        if st.button("🔄 Procesar PDF"):
            with st.spinner("📖 Extrayendo texto del PDF..."):
                try:
                    chunks = extract_text_from_pdf(
                        pdf_path="temp_pdf.pdf",
                        use_blocks=use_blocks,
                        min_length=min_length,
                        exclude_headers_footers=exclude_headers_footers,
                        header_height=header_height,
                        footer_height=footer_height,
                        start_at_page=start_page,
                        stop_at_page=end_page,
                        omit_texts=omit_texts
                    )
                    
                    st.success(f"✅ Extraídos {len(chunks)} chunks de texto")
                    
                    # Mostrar algunos chunks de ejemplo
                    with st.expander("📄 Ver chunks extraídos"):
                        for i, chunk in enumerate(chunks[:5]):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(chunk)
                            st.divider()
                    
                    # Guardar chunks
                    if st.button("💾 Guardar Chunks"):
                        with open("chunks_v2.json", "w", encoding="utf-8") as f:
                            json.dump(chunks, f, ensure_ascii=False, indent=2)
                        st.success("✅ Chunks guardados en chunks_v2.json")
                    
                    # Generar embeddings
                    if st.button("🧠 Generar Embeddings"):
                        if len(chunks) > 0:
                            embeddings = embed_texts_openai(chunks, client)
                            if len(embeddings) > 0:
                                save_embeddings_to_pkl(embeddings, chunks, "embeddings_guia_alimentaria_v2.pkl")
                                st.success("✅ Embeddings generados y guardados")
                                st.rerun()
                        else:
                            st.error("No hay chunks para procesar")
                
                except Exception as e:
                    st.error(f"Error procesando PDF: {e}")
        
        # Limpiar archivo temporal
        if os.path.exists("temp_pdf.pdf"):
            os.remove("temp_pdf.pdf")

with tab3:
    st.header("ℹ️ Información del Sistema")
    
    st.subheader("🎯 ¿Qué es este sistema?")
    st.write("""
    Este es un chatbot especializado en nutrición durante el embarazo que utiliza tecnología de **LLM (Large Language Models) + RAG (Retrieval-Augmented Generation)**.
    
    **Características principales:**
    - 🤖 Utiliza modelos de lenguaje avanzados de OpenAI
    - 🔍 Sistema de recuperación de información semántica con FAISS
    - 📚 Base de conocimiento en guía alimentaria para mujeres embarazadas
    - 💬 Interfaz conversacional intuitiva
    - 🧠 Memoria de conversación con resúmenes automáticos
    """)
    
    st.subheader("🔧 Tecnologías utilizadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Frontend:**
        - Streamlit (interfaz web)
        
        **Procesamiento:**
        - PyMuPDF (extracción de PDF)
        - OpenAI API (embeddings y LLM)
        - FAISS (búsqueda semántica)
        """)
    
    with col2:
        st.markdown("""
        **Almacenamiento:**
        - JSON (chunks de texto)
        - Pickle (embeddings)
        
        **Modelos:**
        - text-embedding-3-small (embeddings)
        - GPT-4o-mini (generación de respuestas)
        """)
    
    st.subheader("📋 Funcionalidades")
    st.write("""
    1. **Extracción de texto:** Procesa PDFs de guías alimentarias
    2. **Generación de embeddings:** Convierte texto en vectores numéricos
    3. **Búsqueda semántica:** Encuentra información relevante
    4. **Generación de respuestas:** Crea respuestas contextualizadas
    5. **Memoria de conversación:** Mantiene contexto de la charla
    6. **Interfaz web:** Fácil de usar desde cualquier navegador
    """)
    
    st.subheader("👥 Equipo de Desarrollo")
    st.write("""
    **Equipo 12 - Maestría en Inteligencia Artificial Aplicada**
    - Fernando Omar Salazar Ortiz - A01796214
    - Carlos Aaron Bocanegra Buitron - A01796345
    - Luis Enrique González González - A01795338
    - Gloria María Campos García - A01422345
    
    **Tecnológico de Monterrey**
    **Profesor:** Luis Eduardo Falcón Morales
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤰 Chatbot de Nutrición para Embarazadas | LLM + RAG | Equipo 12</p>
        <p>Desarrollado con ❤️ para mejorar la nutrición materna</p>
    </div>
    """,
    unsafe_allow_html=True
) 