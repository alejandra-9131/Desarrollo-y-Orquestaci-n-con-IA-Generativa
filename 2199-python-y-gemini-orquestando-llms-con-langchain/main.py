from langchain.agents import AgentExecutor
from orquestador import AgenteOrquestador
from langchain.agents import Tool


def main():
    agente = AgenteOrquestador()
    ejecutor = AgentExecutor(
        agent=agente.agente,
        tools=agente.tools,
        verbose=True
    )
    pregunta = "Quiero que me expliques como funcionan los desvios condicionales?"

    respuesta = ejecutor.invoke({"input": pregunta})

    print(respuesta)

if __name__ == "__main__":
    main()