import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.docstore.document import Document

# --- Configuración y Carga ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# Conexión a Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Modelos de IA
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("la variable de entorno GOOGLE_API_KEY no está configurada.")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

print("¡Servidor de Noticias listo!")

# --- Endpoint Principal ---
@app.route('/ask', methods=['POST'])
def ask_question():
    json_data = request.get_json()
    pregunta = json_data.get('question', '')
    if not pregunta:
        return jsonify({"error": "No se proporcionó ninguna pregunta."}), 400

    print(f"Recibida pregunta: '{pregunta}'")

    try:
        # 1. Convertimos la pregunta del usuario en un "embedding"
        query_embedding = embeddings.embed_query(pregunta)

        # 2. Buscamos en Supabase los artículos más relevantes
        # Usamos la función que creamos en el SQL Editor
        matching_articles = supabase.rpc('match_articles', {
            'query_embedding': query_embedding,
            'match_threshold': 0.7, # Umbral de similitud
            'match_count': 5        # Número de artículos a traer
        }).execute()
        
        if not matching_articles.data:
            return jsonify({"answer": "No encontré noticias relevantes sobre ese tema en mi base de datos.", "sources": []})

        # 3. Creamos el contexto con la información encontrada
        context_text = ""
        sources = []
        for article in matching_articles.data:
            context_text += f"Fuente: {article['title']}\nContenido: {article['content']}\n\n"
            sources.append({"title": article['title'], "url": article['url']})

        # 4. Creamos el prompt y la cadena para generar la respuesta
        prompt_template = """
        Eres un asistente de IA que responde preguntas basándose ÚNICAMENTE en las noticias proporcionadas.
        Contexto de las noticias:
        {context}

        Pregunta del usuario: {question}

        Respuesta en español:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = LLMChain(llm=llm, prompt=PROMPT)
        
        # 5. Generamos la respuesta
        respuesta_texto = chain.invoke({
            "context": context_text,
            "question": pregunta
        })['text']

        return jsonify({"answer": respuesta_texto, "sources": sources})

    except Exception as e:
        print(f"Error en el servidor de noticias: {e}")
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

if __name__ == '__main__':
    # Lo ejecutaremos en un puerto diferente para no chocar con el otro servidor
    app.run(host='0.0.0.0', port=8081)