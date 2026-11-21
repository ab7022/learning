"""
agent_loop.py - A minimal, readable example of how an AI "agent loop" works.

THE IDEA IN ONE SENTENCE
------------------------
An agent is just a loop: send the conversation + a list of tools to the LLM,
and if the LLM asks to use a tool, run that tool, append the result to the
conversation, and ask the LLM again - until it answers with plain text.

    user question
         |
         v
    +-----------+   "call tool X with args"   +--------------+
    |    LLM    | --------------------------> | run tool X   |
    |           | <-------------------------- | (your code)  |
    +-----------+      tool result            +--------------+
         |
         | (no more tool calls)
         v
    final answer

HOW TO RUN
----------
    python agent_loop.py                 # uses a fake "mock" LLM, no API key needed
    pip install anthropic
    ANTHROPIC_API_KEY=sk-... python agent_loop.py --real   # uses a real Claude model
"""

import json
import math
import sys
from datetime import datetime


# =============================================================================
# STEP 1: Define tools as plain Python functions
# =============================================================================
# A "tool" is nothing magical - it's a normal function. The LLM never runs it;
# it only *asks* us to run it. We run it and send back the result.

def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 3 * 4' or 'sqrt(16)'."""
    allowed = {"sqrt": math.sqrt, "pi": math.pi, "pow": pow}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Pretend weather lookup (a real one would call a weather API)."""
    fake_data = {"delhi": "31°C, sunny", "mumbai": "29°C, humid", "london": "12°C, rainy"}
    return fake_data.get(city.lower(), f"No weather data for {city}")


def get_time() -> str:
    """Return the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# STEP 2: Describe the tools to the LLM (the "schema")
# =============================================================================
# The LLM can't read your Python code. It only sees a name, a description,
# and a JSON schema of the inputs. Good descriptions = the LLM picks the right tool.
# To ADD A NEW TOOL: write a function above, add a schema here, add it to TOOL_FUNCTIONS.

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression. Use for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '12 * 7'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# Maps the tool name the LLM sends back -> the Python function to run.
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_time": get_time,
}


# =============================================================================
# STEP 3: Run a tool when the LLM asks for it (the "dispatcher")
# =============================================================================

def run_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'"
    try:
        return func(**args)          # e.g. calculator(expression="12 * 7")
    except Exception as e:
        return f"Error running {name}: {e}"   # errors go back to the LLM too


# =============================================================================
# STEP 4: The LLM
# =============================================================================
# Every LLM call returns one of two things:
#   (a) a final text answer          -> stop_reason == "end_turn"
#   (b) one or more tool requests    -> stop_reason == "tool_use"
# Both the mock and the real client below return the same shape so the loop
# doesn't care which one it's talking to.

class MockLLM:
    """A fake LLM with hard-coded 'reasoning', so you can run this offline."""

    def create(self, messages, tools):
        last = messages[-1]

        # If the last message holds tool results, the "LLM" now writes the answer.
        if isinstance(last["content"], list) and last["content"][0].get("type") == "tool_result":
            results = [block["content"] for block in last["content"]]
            return {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Here's what I found: " + " | ".join(results)}],
            }

        # Otherwise, decide which tools to call based on the question.
        question = last["content"].lower()
        calls = []
        if "weather" in question:
            calls.append({"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Delhi"}})
        if any(op in question for op in ["+", "*", "times", "plus", "calculate"]):
            calls.append({"type": "tool_use", "id": "call_2", "name": "calculator", "input": {"expression": "23 * 17"}})
        if "time" in question and "times" not in question:
            calls.append({"type": "tool_use", "id": "call_3", "name": "get_time", "input": {}})

        if calls:
            return {"stop_reason": "tool_use", "content": calls}
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "I can answer that without tools!"}]}


class ClaudeLLM:
    """A thin wrapper around the real Anthropic API."""

    def __init__(self, model="claude-sonnet-5-5"):
        import anthropic
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        self.model = model

    def create(self, messages, tools):
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, tools=tools, messages=messages
        )
        return {"stop_reason": resp.stop_reason, "content": [b.model_dump() for b in resp.content]}


# =============================================================================
# STEP 5: THE AGENT LOOP  (this is the important part)
# =============================================================================

def agent_loop(user_question: str, llm, max_steps: int = 10) -> str:
    # The conversation is just a growing list of messages.
    messages = [{"role": "user", "content": user_question}]

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}: asking the LLM ---")
        response = llm.create(messages=messages, tools=TOOLS)

        # Always save what the LLM said, so it remembers its own tool requests.
        messages.append({"role": "assistant", "content": response["content"]})

        # (a) No tool requested -> the LLM is done. Return its text.
        if response["stop_reason"] != "tool_use":
            answer = "".join(b["text"] for b in response["content"] if b["type"] == "text")
            print("LLM gave a final answer.")
            return answer

        # (b) The LLM chose one or more tools. Run each and collect results.
        tool_results = []
        for block in response["content"]:
            if block["type"] != "tool_use":
                continue  # LLMs sometimes "think out loud" in text before a tool call
            print(f"LLM chose tool: {block['name']}({json.dumps(block['input'])})")
            result = run_tool(block["name"], block["input"])
            print(f"  -> tool returned: {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],   # links this result to the request
                "content": result,
            })

        # Send the results back as a "user" message, then loop: the LLM sees
        # the results and decides whether to call more tools or answer.
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: too many steps (safety limit reached)."


# =============================================================================
# Try it
# =============================================================================

if __name__ == "__main__":
    llm = ClaudeLLM() if "--real" in sys.argv else MockLLM()
    print(f"Using: {type(llm).__name__}")

    for question in [
        "What's the weather in Delhi, and what is 23 times 17?",
        "What time is it?",
        "Say hello!",
    ]:
        print("\n" + "=" * 60)
        print(f"USER: {question}")
        print("\nFINAL ANSWER:", agent_loop(question, llm))
