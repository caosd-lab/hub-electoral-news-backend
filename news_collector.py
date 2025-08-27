import os
import time
from bs4 import BeautifulSoup
from supabase import create_client, Client
from dotenv import load_dotenv
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
# <<< 1. IMPORTAMOS LAS HERRAMIENTAS DE EMBEDDING >>>
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Cargar las "llaves" desde nuestro archivo .env
load_dotenv()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# <<< 2. INICIALIZAMOS EL MODELO DE EMBEDDINGS >>>
# Asegúrate de que tu GOOGLE_API_KEY esté en el archivo .env o la hayas exportado
if "GOOGLE_API_KEY" not in os.environ:
    # Si no encuentra la clave, la pedirá por la terminal para que no se detenga.
    os.environ["GOOGLE_API_KEY"] = input("Por favor, pega tu GOOGLE_API_KEY y presiona Enter: ")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


print("Conectado a Supabase.")

def scrape_emol():
    """
    Esta función usa un navegador real (vía Playwright) para visitar Emol,
    extraer artículos, calcular sus embeddings y guardarlos en la base de datos.
    """
    with sync_playwright() as p:
        # Lanzamos un navegador (en modo "headless", es decir, invisible)
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Visitamos la página principal de la sección
        site_url = "https://www.emol.com/tag/elecciones-2025/353/todas.aspx"
        print(f"Visitando {site_url} con un navegador real...")
        page.goto(site_url, wait_until="domcontentloaded")

        # Esperamos un poco para que cargue contenido dinámico
        time.sleep(3)

        # Obtenemos el HTML después de que el JavaScript se haya ejecutado
        html_content = page.content()
        soup = BeautifulSoup(html_content, 'html.parser')

        # 2. Buscamos los titulares con el selector que ya encontramos
        # Nota: El selector 'cont_bus_txt_detall_2' es el que tú encontraste. ¡Perfecto!
        headlines = soup.find_all('div', class_='cont_bus_txt_detall_2')
        print(f"Se encontraron {len(headlines)} titulares. Procesando...")

        for headline in headlines:
            try:
                article_link_tag = headline.find('a')
                if not article_link_tag:
                    continue

                article_url = urljoin("https://www.emol.com/", article_link_tag['href'])
                article_title = article_link_tag.text.strip()
                
                existing_article = supabase.table('articles').select('id').eq('url', article_url).execute()
                if existing_article.data:
                    print(f"El artículo '{article_title}' ya existe. Saltando.")
                    continue

                # 4. Visitamos la página del artículo con nuestro navegador
                print(f"  > Visitando artículo: {article_title}")
                page.goto(article_url, wait_until="domcontentloaded")
                time.sleep(3) # Damos tiempo para que cargue el contenido del artículo

                article_html = page.content()
                article_soup = BeautifulSoup(article_html, 'html.parser')

                # Buscamos el div que contiene el cuerpo del artículo
                # Nota: El selector 'cuDetalle_cuTexto_textoNoticia' es el que tú encontraste. ¡Perfecto!
                article_body = article_soup.find('div', id='cuDetalle_cuTexto_textoNoticia')
                
                if article_body:
                    article_content = article_body.get_text(separator="\n", strip=True)
                else:
                    article_content = "" # No se encontró el cuerpo del artículo

                if article_content:
                    print(f"  > Contenido extraído. Calculando embedding...")
                    # <<< 3. CALCULAMOS LA "HUELLA DIGITAL" DEL CONTENIDO >>>
                    embedding = embeddings.embed_query(article_content)
                    
                    print(f"  > Guardando en la base de datos (con embedding)...")
                    # <<< 4. GUARDAMOS EL NUEVO CAMPO 'embedding' >>>
                    data, count = supabase.table('articles').insert({
                        'source': 'Emol',
                        'title': article_title,
                        'url': article_url,
                        'content': article_content,
                        'embedding': embedding
                    }).execute()
                else:
                    print("  > No se encontró contenido extraíble en el artículo.")

            except Exception as e:
                print(f"  > Error procesando un artículo: {e}")
        
        # Cerramos el navegador al final
        browser.close()

# --- Ejecutamos nuestro recolector ---
if __name__ == "__main__":
    scrape_emol()
    print("\nProceso de recolección finalizado.")