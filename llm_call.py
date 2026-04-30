import ollama


def call_llm(prompt, model="gemma3:4b"):
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return (resp["message"]["content"] or "").strip()
