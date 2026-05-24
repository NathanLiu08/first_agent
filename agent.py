from pathlib import Path
from agents import Agent, Runner, function_tool


CONCERT_FILE = Path(__file__).parent / "concert.txt"


def load_concert_info() -> str:
    """
    Read concert information from concert.txt.
    """
    if not CONCERT_FILE.exists():
        return (
            "ERROR: concert.txt was not found. "
            "Please put concert.txt in the same folder as agent.py."
        )

    return CONCERT_FILE.read_text(encoding="utf-8")


@function_tool
def search_concert_info() -> str:
    """
    Read Saratoga High School concert information from concert.txt.
    Use this tool whenever the user asks about Saratoga High School concerts,
    music events, dates, times, locations, tickets, performers, or schedules.
    """
    return load_concert_info()


agent = Agent(
    name="Saratoga High School Concert Assistant",
    instructions=(
        "You are a helpful assistant that answers questions about Saratoga High School "
        "and its concert information. "
        "When the user asks about concerts, music events, schedules, dates, times, "
        "locations, tickets, performers, or related school concert details, you must use "
        "the search_concert_info tool. "
        "Answer using only the information from concert.txt when the question is about "
        "concert information. "
        "If the answer is not in concert.txt, say you could not find that information "
        "in the file. Do not make up details."
    ),
    tools=[search_concert_info],
)


def main():
    print("Saratoga High School Concert Assistant")
    print("Ask me about Saratoga High School concert information.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = Runner.run_sync(agent, question)
        print("\nAgent:", result.final_output)
        print()


if __name__ == "__main__":
    main()