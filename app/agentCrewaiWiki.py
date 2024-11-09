from crewai import Agent, Task, Crew
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import requests
from dotenv import load_dotenv
load_dotenv()

# Schemas para los argumentos de las herramientas
class WikipediaSearchArgs(BaseModel):
    query: str = Field(..., description="El término exacto que se desea buscar en Wikipedia")

class SaveFileArgs(BaseModel):
    texto: str = Field(..., description="El contenido que se desea guardar en el archivo")

# Definimos las funciones para las herramientas
def buscar_wikipedia(query: str) -> str:
    """Busca información en Wikipedia."""
    try:
        response = requests.get(f"https://es.wikipedia.org/api/rest_v1/page/summary/{query}")
        if response.status_code == 200:
            data = response.json()
            return data.get("extract") or "No se encontró información sobre este tema en Wikipedia."
        else:
            return "Error al consultar Wikipedia."
    except Exception as e:
        return f"Error al realizar la búsqueda: {str(e)}"

def guardar_en_archivo(texto: str) -> str:
    """Guarda el texto en un archivo."""
    try:
        with open("informe.txt", "w", encoding="utf-8") as file:
            file.write(texto)
        return f"El informe ha sido guardado en informe.txt"
    except Exception as e:
        return f"Error al guardar el archivo: {str(e)}"

# Creamos las herramientas con sus schemas
tool_wikipedia = Tool.from_function(
    name="Buscar_en_Wikipedia",
    description="Busca información en Wikipedia sobre un tema específico",
    func=buscar_wikipedia,
    args_schema=WikipediaSearchArgs
)

tool_guardar = Tool.from_function(
    name="Guardar_en_Archivo",
    description="Guarda el texto proporcionado en un archivo llamado 'informe.txt'",
    func=guardar_en_archivo,
    args_schema=SaveFileArgs
)

# Configuración del modelo de lenguaje 
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

# Creación de los agentes con sus herramientas
investigador = Agent(
    role='Investigador Wikipedia',
    goal='Buscar y obtener información detallada de Wikipedia',
    backstory="""Eres un investigador experto en buscar información en Wikipedia.
    Tu trabajo es usar la herramienta de búsqueda para obtener información precisa y detallada.""",
    tools=[tool_wikipedia],
    llm=llm,
    verbose=True
)

escritor = Agent(
    role='Escritor y Archivador',
    goal='Crear resúmenes concisos y guardar información en archivos',
    backstory="""Eres un escritor experto en crear resúmenes concisos y guardar información.
    Tu trabajo es resumir la información y usar la herramienta de guardado para almacenarla.""",
    tools=[tool_guardar],
    llm=llm,
    verbose=True
)

# Creación de las tareas
tarea_busqueda = Task(
    description="""Busca información sobre {query} en Wikipedia.
    Utiliza la herramienta Buscar_en_Wikipedia y asegúrate de obtener información relevante.""",
    expected_output="Información detallada obtenida de Wikipedia sobre {query}",
    agent=investigador
)

tarea_resumen = Task(
    description="""Crea un resumen de la información obtenida sobre {query} y guárdalo en un archivo.
    El resumen debe ser claro, conciso y contener los puntos más importantes.""",
    expected_output="Un archivo creado con un resumen bien estructurado sobre {query}",
    agent=escritor,
    context=[tarea_busqueda]
)

# Creación y configuración del crew
crew = Crew(
    agents=[investigador, escritor],
    tasks=[tarea_busqueda, tarea_resumen],
    verbose=True
)

def main():
    """Función principal para ejecutar el crew"""
    resultado = crew.kickoff(inputs={"query": "Albert Einstein"})
    print("\nResultado final:")
    print(resultado)

if __name__ == "__main__":
    main()