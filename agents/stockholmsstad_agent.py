# stockholmsstad_agent.py
from util.tools import search_stockholms_stad


SYSTEM_PROMPT = """
Du är en hjälpsam assistent som svarar på frågor om Stockholms stads webbplats.

Regler:
1. När frågan gäller fakta från Stockholms stads webb, använd verktyget search_stockholms_stad.
2. Svara primärt utifrån verktygets resultat.
3. Hitta inte på information om källorna är svaga eller tomma.
4. Ange gärna käll-URL:er kort i svaret.
5. Håll svaret tydligt och konkret.
"""


TOOLS = [
    {
        "name": "search_stockholms_stad",
        "description": "Söker information på Stockholms stads webbplats med enkel RAG över interna sidor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Frågan eller sökuttrycket."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Antal textutdrag att returnera.",
                    "default": 5
                },
                "refresh": {
                    "type": "boolean",
                    "description": "Bygg om index från webbplatsen innan sökning.",
                    "default": False
                }
            },
            "required": ["query"]
        },
        "func": search_stockholms_stad,
    }
]


def run_agent(user_question: str, llm):
    """
    Enkel pseudo-kod för agent-loop.
    Byt ut mot er faktiska LLM/agentintegration.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    # Exempel: om ni själva styr tool-anrop enkelt:
    tool_result = search_stockholms_stad(query=user_question, top_k=5, refresh=False)

    messages.append({
        "role": "tool",
        "name": "search_stockholms_stad",
        "content": str(tool_result),
    })

    response = llm.invoke(messages)
    return response