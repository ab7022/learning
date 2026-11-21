# learning

This is where I keep everything I'm learning: small, simple examples that each explain one idea. I'll keep adding to it as I learn new things.

## Contents

| File | What it shows |
|------|---------------|
| [agent_loop.py](agent_loop.py) | How an AI agent loop works: the LLM picks a tool, the code runs it, and the result goes back to the LLM until it gives a final answer. |

## Running the examples

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key
python agent_loop.py
```
