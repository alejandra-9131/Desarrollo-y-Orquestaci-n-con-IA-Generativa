from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from langchain_core.globals import set_debug
from langchain import hub
from herramienta_analisis_imagen import HerramientaAnalisisImagen
from langchain.agents import Tool, create_react_agent, AgentExecutor
from herramienta_explicar import HerramientaExplicar

set_debug(False)


class AgenteOrquestador:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_FLASH
        )
        herramienta_analisis_imagen = HerramientaAnalisisImagen()
        herramienta_explicar = HerramientaExplicar()

        self.tools = [
            Tool(
                name=herramienta_analisis_imagen.name,
                func=herramienta_analisis_imagen.run,
                description=herramienta_analisis_imagen.description,
                return_direct=herramienta_analisis_imagen.return_direct
            ),
            Tool(
                name=herramienta_explicar.name,
                func=herramienta_explicar.run,
                description=herramienta_explicar.description,
                return_direct=herramienta_explicar.return_direct
            )
        ]
        prompt = hub.pull("hwchase17/react")

        self.agente = create_react_agent(self.llm, self.tools, prompt)

        self.agente_executor = AgentExecutor(
            agent=self.agente,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, input):
        return self.agente_executor.invoke({"input": input})