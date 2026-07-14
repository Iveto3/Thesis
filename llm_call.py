import time
import ollama


def call_llm(prompt, model="gemma3:4b", max_retries=3):
    """ Calls the LLM with a given prompt and returns the response."""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_ctx": 4096,
                    "num_predict": 700,
                },
            )
            return (resp["message"]["content"] or "").strip()

        except Exception as error:
            last_error = error
            print(
                f"[LLM ERROR] attempt {attempt}/{max_retries}: {error}",
                flush=True,
            )
            time.sleep(10 * attempt)

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts: {last_error}")
