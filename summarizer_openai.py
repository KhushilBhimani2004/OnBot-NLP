from openai import OpenAI
import os
import textwrap
import time

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"
CHUNK_CHAR_SIZE = 3500
PAUSE_BETWEEN_REQUESTS = 0.4


def _call_chat_completion(messages, max_tokens=500, temperature=0.4):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def summarize_long_text_with_chunks(text, system_prompt="You are an expert technical summarizer."):
    if not text or not text.strip():
        return "No text to summarize."

    text = " ".join(text.split())
    chunks = textwrap.wrap(text, CHUNK_CHAR_SIZE)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize the following portion of a file in 3-5 concise bullet points:\n\n{chunk}"}
        ]
        try:
            summary = _call_chat_completion(messages, max_tokens=400)
            chunk_summaries.append(f"Chunk {i+1} summary:\n{summary}")
        except Exception as e:
            chunk_summaries.append(f"Chunk {i+1} summary: (error) {e}")
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    combined = "\n\n".join(chunk_summaries)
    final_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Combine the following chunk summaries into one coherent summary of the file (3–5 sentences) and add 3 key takeaways:\n\n" + combined}
    ]
    try:
        final_summary = _call_chat_completion(final_messages, max_tokens=600)
    except Exception as e:
        final_summary = f"Error creating final summary: {e}\n\nPartial summaries:\n{combined[:2000]}"
    return final_summary
