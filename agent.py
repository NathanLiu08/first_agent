from agents import Agent, Runner, function_tool


@function_tool
def get_game_tip(topic: str) -> str:
    """
    Return a simple game development tip.
    """
    tips = {
        "snake": "For Snake, keep the game loop simple: update direction, move the snake, check collision, then redraw.",
        "aws": "For a simple web game, deploy the static files with AWS Amplify or S3 + CloudFront.",
        "javascript": "Keep game state in one object so debugging is easier.",
    }

    return tips.get(topic.lower(), "Keep your first version simple, then improve one feature at a time.")


agent = Agent(
    name="Demo Helper Agent",
    instructions=(
        "You are a helpful coding assistant. "
        "Explain things simply and help beginners build small projects."
    ),
    tools=[get_game_tip],
)


result = Runner.run_sync(
    agent,
    "I made a snake game in JavaScript. Give me one tip about deploying it to AWS."
)

print(result.final_output)