"""LLM-as-judge implementations for eval scoring."""

from openai import OpenAI

client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"

GROUNDEDNESS_PROMPT = """\
You are a strict fact-checker. Given a CONTEXT and an ANSWER, decide whether
the ANSWER relies ONLY on facts present in the CONTEXT.

If the answer introduces any fact not supported by the context, reply NO.
Otherwise reply YES.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with exactly one word: YES or NO."""


def judge_grounded(answer: str, context: str) -> bool:
    """Return True if `answer` is fully grounded in `context`.

    If no context is provided, assume the answer doesn't need grounding.
    """
    if not context:
        return True

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "user",
                "content": GROUNDEDNESS_PROMPT.format(
                    context=context,
                    answer=answer,
                ),
            }
        ],
        temperature=0,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("YES")
