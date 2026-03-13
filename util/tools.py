import math
from datetime import datetime, timezone

from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The name of the city to get weather for.
    """
    # Simulated weather data for demo purposes
    weather_data = {
        "new york": "Partly cloudy, 18°C / 64°F, humidity 55%",
        "london": "Overcast, 12°C / 54°F, humidity 78%",
        "tokyo": "Clear skies, 22°C / 72°F, humidity 40%",
        "paris": "Light rain, 14°C / 57°F, humidity 82%",
        "sydney": "Sunny, 26°C / 79°F, humidity 35%",
    }
    key = city.lower().strip()
    if key in weather_data:
        return f"Weather in {city}: {weather_data[key]}"
    return f"Weather in {city}: Sunny, 20°C / 68°F, humidity 50% (simulated)"


@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: The search query string.
    """
    # Simulated search results for demo purposes
    return (
        f'Search results for "{query}":\n'
        f"1. {query} - Wikipedia overview and key facts\n"
        f"2. Latest news and developments about {query}\n"
        f"3. In-depth analysis: Understanding {query}"
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate.
                    Supports +, -, *, /, **, sqrt(), abs(), etc.
    """
    # Safe math evaluation with limited builtins
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time in UTC."""
    now = datetime.now(timezone.utc)
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
