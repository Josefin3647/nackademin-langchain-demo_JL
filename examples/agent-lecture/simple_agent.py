from langchain.agents import create_agent
from util.tools import get_current_time, calculate
from langgraph.checkpoint.memory import InMemorySaver

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

def run():
    model = get_model(temperature=0.2)
    memory = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_current_time, calculate],
        system_prompt=(
            "Du är en hjälpsam assistent som svarar på användarens frågor. "
            "Svara alltid på svenska och var koncis men informativ."
            "När ett verktyg behövs ska du använda det."
        ),
        checkpointer=memory,
    )

    thread_id = "default"

    print("Chatten är igång. Skriv 'exit', 'quit' eller 'sluta' för att avsluta.\n")

    while True:
        user_input = get_user_input("Ställ din fråga")

        if user_input.lower() in ["exit", "quit", "sluta"]:
            print("Avslutar chatten.")
            break

        process_stream = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode=STREAM_MODES,
            config={"configurable": {"thread_id": thread_id}},
        )

        handle_stream(process_stream)


if __name__ == "__main__":
    run()