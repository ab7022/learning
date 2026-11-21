"""
agent_loop.py - How an agent loop works, in the simplest form.

The loop:
  1. Send the conversation + the list of tools to the LLM.
  2. If the LLM asks to use a tool -> run it, add the result to the conversation, go to 1.
  3. If the LLM replies with plain text -> that's the final answer.

Run:
  pip install anthropic
  export ANTHROPIC_API_KEY=your-key
  python agent_loop.py
"""

import anthropic

client = anthropic.Anthropic()


# 1. The tool: a normal Python function. The LLM never runs it - we do.
def get_weather(city):
    weather = {"delhi": "31°C, sunny", "mumbai": "29°C, humid", "london": "12°C, rainy"}
    return weather.get(city.lower(), "No data for " + city)


# 2. Describe the tool to the LLM. It only sees this, not the code above.
#    To add more tools, add more entries to this list (and to the if-check below).
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


# 3. The agent loop.
messages = [{"role": "user", "content": "What's the weather in Delhi and London?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    # Save the LLM's reply so it remembers what it asked for.
    messages.append({"role": "assistant", "content": response.content})

    # No tool requested -> the LLM is done.
    if response.stop_reason != "tool_use":
        print("Final answer:", response.content[0].text)
        break

    # The LLM chose a tool -> run it and send back the result.
    results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"LLM called {block.name} with {block.input}")

            if block.name == "get_weather":
                output = get_weather(**block.input)

            print("  result:", output)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # links the result to the request
                "content": output,
            })

    messages.append({"role": "user", "content": results})
