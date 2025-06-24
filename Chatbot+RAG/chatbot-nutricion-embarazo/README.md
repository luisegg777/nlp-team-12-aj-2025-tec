# 🤰 Chatbot de Nutrición para Embarazadas - LLM + RAG

## Descripción

Esta aplicación web desarrollada con Streamlit implementa un sistema de chatbot especializado en nutrición durante el embarazo utilizando tecnología de **LLM (Large Language Models) + RAG (Retrieval-Augmented Generation)**. El sistema permite a las mujeres embarazadas hacer consultas sobre alimentación basándose en una guía alimentaria oficial.

## 🚀 Características Principales

- **🤖 Chatbot Inteligente**: Utiliza modelos avanzados de OpenAI para generar respuestas precisas
- **🔍 Búsqueda Semántica**: Sistema RAG con FAISS para recuperar información relevante
- **📚 Base de Conocimiento**: Información extraída de guías alimentarias oficiales
- **💬 Interfaz Conversacional**: Chat intuitivo con memoria de conversación
- **🧠 Memoria Inteligente**: Resúmenes automáticos para mantener contexto
- **📊 Procesamiento de PDFs**: Extracción automática de información de documentos

## 🛠️ Tecnologías Utilizadas

### Frontend
- **Streamlit**: Interfaz web interactiva y responsive

### Backend
- **OpenAI API**: Modelos de lenguaje y embeddings
- **PyMuPDF**: Extracción de texto desde PDFs
- **FAISS**: Búsqueda semántica de vectores
- **NumPy**: Procesamiento numérico

### Almacenamiento
- **JSON**: Chunks de texto extraídos
- **Pickle**: Embeddings vectoriales

## 📦 Instalación

### Prerrequisitos
- Python 3.8 o superior
- API Key de OpenAI

### Pasos de instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd chatbot-nutricion-embarazo
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar API Key**
   - Obtén tu API Key de OpenAI desde [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Configúrala en la aplicación web

## 🚀 Uso

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Flujo de uso

1. **Configuración inicial**:
   - Ingresa tu API Key de OpenAI en la barra lateral
   - Ajusta los parámetros del modelo según tus necesidades

2. **Procesamiento de datos** (primera vez):
   - Ve a la pestaña "Procesamiento"
   - Sube el archivo PDF de la guía alimentaria
   - Configura los parámetros de extracción
   - Procesa el PDF para generar embeddings

3. **Chat**:
   - Ve a la pestaña "Chat"
   - Haz preguntas sobre nutrición durante el embarazo
   - El sistema responderá basándose en la información de la guía

## 📋 Funcionalidades

### 💬 Chat Principal
- **Preguntas y respuestas**: Interfaz conversacional intuitiva
- **Ejemplos de preguntas**: Sugerencias para facilitar el uso
- **Historial de conversación**: Mantiene el contexto de la charla
- **Resúmenes automáticos**: Optimiza el uso de tokens

### 📊 Procesamiento de Datos
- **Carga de PDFs**: Subida de documentos de guías alimentarias
- **Configuración flexible**: Parámetros personalizables para extracción
- **Filtrado inteligente**: Eliminación automática de contenido irrelevante
- **Generación de embeddings**: Conversión de texto a vectores semánticos

### ⚙️ Configuración Avanzada
- **Modelos de lenguaje**: Selección entre diferentes modelos de OpenAI
- **Parámetros RAG**: Ajuste del número de chunks recuperados
- **Visualización de chunks**: Opción para ver la información utilizada
- **Modo sin RAG**: Respuestas basadas solo en el conocimiento del modelo

## 🎯 Ejemplos de Preguntas

El chatbot puede responder preguntas como:

- ¿Puedo comer frutos rojos durante el embarazo?
- ¿Qué alimentos debo evitar durante el embarazo?
- ¿Cuánta agua debo consumir diariamente?
- ¿Puedo comer pescado durante el embarazo?
- ¿Qué frutas son recomendadas durante el embarazo?
- ¿Se puede consumir café durante el embarazo?
- ¿Cuántas porciones de carne debo comer?

## 🔧 Configuración Avanzada

### Parámetros del Modelo
- **Modelo LLM**: Selecciona entre GPT-4o-mini o GPT-3.5-turbo
- **Top-K**: Número de chunks más relevantes a recuperar (1-10)
- **RAG**: Activar/desactivar el sistema de recuperación aumentada
- **Mostrar chunks**: Visualizar la información utilizada para generar respuestas

### Parámetros de Extracción
- **Bloques visuales**: Extracción por bloques o páginas completas
- **Longitud mínima**: Filtro de chunks por tamaño
- **Encabezados y pies**: Exclusión de elementos no relevantes
- **Rango de páginas**: Especificar páginas a procesar
- **Textos a omitir**: Lista personalizable de contenido a filtrar

## 📁 Estructura del Proyecto

```
chatbot-nutricion-embarazo/
├── app.py                 # Aplicación principal Streamlit
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Documentación
├── chunks_v2.json        # Chunks de texto extraídos (generado)
├── embeddings_guia_alimentaria_v2.pkl  # Embeddings (generado)
└── temp_pdf.pdf          # Archivo temporal (generado)
```

## 🚨 Consideraciones Importantes

### Seguridad
- **API Key**: Nunca compartas tu API Key de OpenAI
- **Datos personales**: La aplicación no almacena información personal
- **Uso responsable**: Respeta los límites de uso de la API

### Limitaciones
- **Dependencia de OpenAI**: Requiere conexión a internet y API Key válida
- **Calidad de datos**: Las respuestas dependen de la calidad del PDF procesado
- **Contexto limitado**: Basado únicamente en la información de la guía alimentaria

### Costos
- **API de OpenAI**: El uso genera costos según el plan de OpenAI
- **Embeddings**: Generación inicial puede ser costosa para PDFs grandes
- **Chat**: Cada pregunta genera costos de API

## 🤝 Contribución

### Equipo de Desarrollo
- **Fernando Omar Salazar Ortiz** - A01796214
- **Carlos Aaron Bocanegra Buitron** - A01796345
- **Luis Enrique González González** - A01795338
- **Gloria María Campos García** - A01422345

### Institución
- **Tecnológico de Monterrey**
- **Maestría en Inteligencia Artificial Aplicada**
- **Profesor:** Luis Eduardo Falcón Morales

## 📚 Referencias

- **RAG (Retrieval-Augmented Generation)**: Aytar, A. Y., et al. (2024)
- **Large Language Models**: Naveed, H., et al. (2024)
- **FAISS**: Douze, M., et al. (2025)
- **Guía Alimentaria**: Secretaría de Salud, INSP, UNICEF (2024)

## 📄 Licencia

Este proyecto es parte de una actividad académica del Tecnológico de Monterrey.

## 🆘 Soporte

Para reportar problemas o solicitar mejoras:
1. Revisa la documentación
2. Verifica la configuración de tu API Key
3. Contacta al equipo de desarrollo

---

**Desarrollado con ❤️ para mejorar la nutrición materna** 