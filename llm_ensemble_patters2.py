"""
LLM Ensemble Patterns — Practical Demonstrations
===================================================
Demonstrates six ensemble strategies using three LLM providers:
  • Anthropic Claude (via anthropic SDK)
  • OpenAI GPT (via openai SDK)
  • Google Gemini (via google-genai SDK)

Requirements:
  pip install anthropic openai google-genai python-dotenv

Environment variables (in .env):
  CLAUDKEY=sk-ant-...          # Anthropic API key
  OPENAI_API_KEY=sk-...        # OpenAI API key
  GEMINI_API_KEY=...           # Google Gemini API key
"""

import os
import re
import json
import time
import asyncio
import random
from collections import Counter
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------
import anthropic
import openai
from google import genai

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("CLAUDKEY"))
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Default models — current as of April 2026
# Adjust to whatever you have access to on your account
CLAUDE_MODEL = "claude-sonnet-4-6"          # Anthropic's latest Sonnet (Feb 2026)
OPENAI_MODEL = "gpt-4.1"                   # OpenAI's latest general-purpose (still available in API)
GEMINI_MODEL = "gemini-2.5-flash"           # Google's best price-performance model

# Smaller / cheaper tiers for routing demo
CLAUDE_SMALL = "claude-haiku-4-5-20251001"  # Anthropic's fastest/cheapest
OPENAI_SMALL = "gpt-4.1-mini"              # OpenAI's efficient small model
GEMINI_SMALL = "gemini-2.5-flash-lite"      # Google's budget tier


# ---------------------------------------------------------------------------
# Unified call wrappers
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """Standardized response container across providers."""
    provider: str
    model: str
    text: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)


def call_claude(prompt: str, system: str = "", model: str = CLAUDE_MODEL, max_tokens: int = 1024) -> LLMResponse:
    """Call Anthropic Claude via the Messages API."""
    t0 = time.perf_counter()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    response = anthropic_client.messages.create(**kwargs)
    latency = (time.perf_counter() - t0) * 1000
    text = response.content[0].text
    return LLMResponse(
        provider="anthropic", model=model, text=text, latency_ms=latency,
        metadata={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    )


def call_openai(prompt: str, system: str = "", model: str = OPENAI_MODEL, max_tokens: int = 1024) -> LLMResponse:
    """Call OpenAI GPT via the Chat Completions API."""
    t0 = time.perf_counter()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = openai_client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    latency = (time.perf_counter() - t0) * 1000
    text = response.choices[0].message.content
    return LLMResponse(
        provider="openai", model=model, text=text, latency_ms=latency,
        metadata={"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens},
    )


def call_gemini(prompt: str, system: str = "", model: str = GEMINI_MODEL, max_tokens: int = 1024) -> LLMResponse:
    """Call Google Gemini via the google-genai SDK."""
    t0 = time.perf_counter()
    config = genai.types.GenerateContentConfig(max_output_tokens=max_tokens)
    if system:
        config.system_instruction = system
    response = gemini_client.models.generate_content(model=model, contents=prompt, config=config)
    latency = (time.perf_counter() - t0) * 1000
    text = response.text
    return LLMResponse(
        provider="gemini", model=model, text=text, latency_ms=latency,
        metadata={"model": model},
    )


# Convenience dispatcher
CALLERS = {
    "claude": call_claude,
    "openai": call_openai,
    "gemini": call_gemini,
}


def call_all(prompt: str, system: str = "", models: Optional[dict] = None) -> list[LLMResponse]:
    """Call all three providers in sequence, return list of responses."""
    models = models or {}
    results = []
    for name, caller in CALLERS.items():
        kwargs = {"prompt": prompt, "system": system}
        if name in models:
            kwargs["model"] = models[name]
        try:
            results.append(caller(**kwargs))
        except Exception as e:
            print(f"  ⚠ {name} failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def header(title: str, emoji: str = "🔷"):
    width = 70
    print(f"\n{'═' * width}")
    print(f" {emoji}  {title}")
    print(f"{'═' * width}\n")


def sub(text: str):
    print(f"  ➤ {text}")


def show_response(resp: LLMResponse, label: str = ""):
    tag = label or f"{resp.provider}/{resp.model}"
    print(f"\n  ┌─ {tag} ({resp.latency_ms:.0f}ms)")
    for line in resp.text.strip().split("\n"):
        print(f"  │ {line}")
    print(f"  └{'─' * 50}")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE 1: Majority Voting
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_majority_vote():
    header("Ensemble 1: Majority Voting", "🗳️")
    print("  Multiple LLMs answer the same classification question.")
    print("  The most common answer wins.\n")

    prompt = (
        "Classify the sentiment of this review as exactly one word — "
        "POSITIVE, NEGATIVE, or NEUTRAL:\n\n"
        "\"The battery life is incredible but the camera quality is mediocre "
        "at best. For the price, I expected much more.\""
    )
    sub(f"Prompt: {prompt[:90]}...")

    responses = call_all(prompt, system="Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.")

    votes = []
    for r in responses:
        # Extract the classification word
        raw = r.text.strip().upper()
        for label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            if label in raw:
                votes.append(label)
                show_response(r, f"{r.provider} → {label}")
                break
        else:
            votes.append(raw)
            show_response(r, f"{r.provider} → {raw} (unparsed)")

    if votes:
        winner, count = Counter(votes).most_common(1)[0]
        total = len(votes)
        print(f"\n  🏆 Majority vote: {winner} ({count}/{total} agreement)")
    else:
        print("\n  ⚠ No valid votes collected.")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE 2: Weighted Ensemble
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_weighted_vote():
    header("Ensemble 2: Weighted Ensemble", "⚖️")
    print("  Each model's vote is weighted by domain-calibrated confidence.")
    print("  Weights would normally be learned from a validation set.\n")

    # Hypothetical weights calibrated on a sentiment benchmark
    WEIGHTS = {"claude": 0.40, "openai": 0.35, "gemini": 0.25}
    print(f"  Weights: {WEIGHTS}\n")

    prompt = (
        "Classify the sentiment of this text as exactly one word — "
        "POSITIVE, NEGATIVE, or NEUTRAL:\n\n"
        "\"I've been using this framework for six months. It has a steep "
        "learning curve but once you get past that, productivity skyrockets.\""
    )
    sub(f"Prompt: {prompt[:90]}...")

    responses = call_all(prompt, system="Respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.")

    weighted_scores: dict[str, float] = {}
    for r in responses:
        raw = r.text.strip().upper()
        label = next((l for l in ["POSITIVE", "NEGATIVE", "NEUTRAL"] if l in raw), raw)
        weight = WEIGHTS.get(r.provider, 0.0)
        weighted_scores[label] = weighted_scores.get(label, 0.0) + weight
        show_response(r, f"{r.provider} (w={weight}) → {label}")

    if weighted_scores:
        winner = max(weighted_scores, key=weighted_scores.get)
        print(f"\n  Weighted scores: {dict(sorted(weighted_scores.items(), key=lambda x: -x[1]))}")
        print(f"  🏆 Weighted winner: {winner} (score={weighted_scores[winner]:.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE 3: Mixture of Agents (MoA)
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_mixture_of_agents():
    header("Ensemble 3: Mixture of Agents (MoA)", "🧬")
    print("  Layer 1: All models answer independently.")
    print("  Layer 2: A synthesizer model sees all Layer 1 outputs and")
    print("           produces a refined, final answer.\n")

    question = "What are the three most important considerations when designing a distributed caching layer for a microservices architecture?"
    sub(f"Question: {question[:80]}...\n")

    # --- Layer 1: Independent responses ---
    print("  ── Layer 1: Independent generation ──")
    layer1 = call_all(question)
    for r in layer1:
        show_response(r, f"Layer1/{r.provider}")

    # --- Layer 2: Synthesis ---
    print("\n  ── Layer 2: Synthesis ──")
    combined = "\n\n".join(
        f"=== Response from {r.provider}/{r.model} ===\n{r.text}" for r in layer1
    )

    synthesis_prompt = (
        f"You are a senior architect synthesizing expert opinions. Below are "
        f"three independent answers to this question:\n\n"
        f"Q: {question}\n\n{combined}\n\n"
        f"Produce a single, authoritative answer that:\n"
        f"1. Integrates the strongest points from each response\n"
        f"2. Resolves any contradictions\n"
        f"3. Adds structure and clarity\n"
        f"Keep it concise (under 300 words)."
    )

    # Use Claude as the synthesizer (typically your strongest reasoning model)
    final = call_claude(synthesis_prompt, system="You are a precise technical synthesizer.", max_tokens=1024)
    show_response(final, "Layer2/synthesizer (Claude)")

    total_latency = sum(r.latency_ms for r in layer1) + final.latency_ms
    print(f"\n  ⏱ Total pipeline latency: {total_latency:.0f}ms")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE 4: LLM Router / Cascade
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_router():
    header("Ensemble 4: LLM Router / Cascade", "🔀")
    print("  A classifier routes each query to the cheapest model that can")
    print("  handle it. Simple → small model, Complex → frontier model.\n")

    queries = [
        "What is 2 + 2?",
        "Translate 'hello' to French.",
        "Explain the CAP theorem and its practical implications for designing "
        "a globally distributed database with strong consistency requirements, "
        "including strategies for handling network partitions.",
    ]

    # Step 1: Use a small/fast model as the router
    def route_query(query: str) -> str:
        """Use a small LLM to classify query difficulty."""
        router_prompt = (
            f"Classify the following query as SIMPLE, MEDIUM, or COMPLEX. "
            f"Respond with exactly one word.\n\nQuery: {query}"
        )
        resp = call_claude(router_prompt, model=CLAUDE_SMALL, max_tokens=10)
        raw = resp.text.strip().upper()
        for level in ["SIMPLE", "MEDIUM", "COMPLEX"]:
            if level in raw:
                return level
        return "MEDIUM"  # default fallback

    # Step 2: Map difficulty to model tier
    TIER_MAP = {
        "SIMPLE":  {"caller": call_gemini,  "model": GEMINI_SMALL, "label": f"Gemini 2.5 Flash-Lite ({GEMINI_SMALL})"},
        "MEDIUM":  {"caller": call_openai,  "model": OPENAI_SMALL, "label": f"GPT-4.1 Mini ({OPENAI_SMALL})"},
        "COMPLEX": {"caller": call_claude,  "model": CLAUDE_MODEL, "label": f"Claude Sonnet 4.6 ({CLAUDE_MODEL})"},
    }

    for i, query in enumerate(queries, 1):
        print(f"  ── Query {i} ──")
        sub(f"Input: {query[:70]}{'...' if len(query) > 70 else ''}")

        difficulty = route_query(query)
        tier = TIER_MAP[difficulty]
        sub(f"Router decision: {difficulty} → {tier['label']}")

        resp = tier["caller"](query, model=tier["model"])
        show_response(resp, f"Routed → {resp.provider}")
        print()

    print("  💡 Cost savings: simple queries used the cheapest model tier,")
    print("     complex queries got the full frontier model.")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE 5: Sequential Chain (Draft → Critique → Polish)
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_sequential_chain():
    header("Ensemble 5: Sequential Chain", "⛓️")
    print("  Stage 1 (Gemini):  Draft a quick answer")
    print("  Stage 2 (OpenAI):  Critique the draft for errors")
    print("  Stage 3 (Claude):  Polish into a final deliverable\n")

    topic = "Write a concise explanation of how a Kubernetes Horizontal Pod Autoscaler works, suitable for a DevOps engineer new to K8s."
    sub(f"Topic: {topic[:80]}...\n")

    # Stage 1: Draft (fast model)
    print("  ── Stage 1: Draft (Gemini) ──")
    draft = call_gemini(topic, system="Write a clear first draft. Be thorough but don't worry about polish.")
    show_response(draft, "Stage1/draft")

    # Stage 2: Critique (reasoning model)
    print("  ── Stage 2: Critique (OpenAI) ──")
    critique_prompt = (
        f"Review this draft for technical accuracy, completeness, and clarity. "
        f"List specific issues and suggest improvements.\n\n"
        f"DRAFT:\n{draft.text}"
    )
    critique = call_openai(critique_prompt, system="You are a meticulous technical reviewer.")
    show_response(critique, "Stage2/critique")

    # Stage 3: Polish (strong writer)
    print("  ── Stage 3: Polish (Claude) ──")
    polish_prompt = (
        f"Given the original draft and reviewer feedback below, produce a "
        f"polished final version. Fix all noted issues, improve clarity, "
        f"and ensure technical accuracy.\n\n"
        f"ORIGINAL DRAFT:\n{draft.text}\n\n"
        f"REVIEWER FEEDBACK:\n{critique.text}"
    )
    final = call_claude(polish_prompt, system="You are an expert technical writer. Produce clean, accurate prose.")
    show_response(final, "Stage3/polished-final")

    total = draft.latency_ms + critique.latency_ms + final.latency_ms
    print(f"\n  ⏱ Total pipeline: {total:.0f}ms across 3 stages")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE 6: Speculative Decoding (Simulated)
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_speculative_decoding():
    header("Ensemble 6: Speculative Decoding (Simulated)", "⚡")
    print("  In real speculative decoding, a small model drafts tokens and a")
    print("  large model verifies them in parallel for 2-3× speedup.")
    print("  Here we simulate the pattern at the sentence level.\n")

    prompt = "Explain in 3-4 sentences how garbage collection works in Python."

    # Step 1: Fast draft from small model
    print("  ── Step 1: Fast draft (small model) ──")
    draft = call_gemini(prompt, model=GEMINI_SMALL, system="Answer concisely in 3-4 sentences.")
    show_response(draft, "Draft/small-model")

    # Step 2: Verification by large model
    print("  ── Step 2: Verification (large model) ──")
    verify_prompt = (
        f"A smaller model produced this draft answer. Your job is to verify it.\n\n"
        f"QUESTION: {prompt}\n"
        f"DRAFT ANSWER: {draft.text}\n\n"
        f"Evaluate each sentence for correctness. Respond with:\n"
        f"- ACCEPT if the draft is fully correct\n"
        f"- REVISE followed by the corrected answer if any part is wrong\n"
        f"Be precise about what, if anything, is incorrect."
    )
    verification = call_claude(verify_prompt, system="You are a rigorous technical verifier.")
    show_response(verification, "Verify/large-model")

    # Determine outcome
    accepted = "ACCEPT" in verification.text.upper()[:50]
    if accepted:
        print("\n  ✅ Draft ACCEPTED — small model output used directly.")
        print(f"  ⚡ Effective latency: {draft.latency_ms:.0f}ms (draft) + {verification.latency_ms:.0f}ms (verify)")
        print(f"     vs ~{verification.latency_ms * 2:.0f}ms if large model generated from scratch")
    else:
        print("\n  🔄 Draft REVISED — large model corrected the output.")
        print(f"  ⏱ Total latency: {draft.latency_ms + verification.latency_ms:.0f}ms")
        print("     (Overhead from failed speculation, but output quality preserved)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "▓" * 70)
    print("  LLM ENSEMBLE PATTERNS — Live Demonstrations")
    print("  Using: Claude Sonnet 4.6 · GPT-4.1 · Gemini 2.5 Flash")
    print("▓" * 70)

    # Verify API keys
    missing = []
    if not os.environ.get("CLAUDKEY"):
        missing.append("CLAUDKEY")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")

    if missing:
        print(f"\n  ⚠ Missing environment variables: {', '.join(missing)}")
        print("  Add them to your .env file and try again.\n")
        return

    demos = [
        ("1", "Majority Voting",          ensemble_majority_vote),
        ("2", "Weighted Ensemble",         ensemble_weighted_vote),
        ("3", "Mixture of Agents",         ensemble_mixture_of_agents),
        ("4", "LLM Router / Cascade",      ensemble_router),
        ("5", "Sequential Chain",          ensemble_sequential_chain),
        ("6", "Speculative Decoding",      ensemble_speculative_decoding),
    ]

    print("\n  Available ensembles:")
    for num, name, _ in demos:
        print(f"    {num}. {name}")
    print(f"    A. Run ALL\n")

    choice = input("  Select ensemble (1-6, A for all): ").strip().upper()

    if choice == "A":
        for _, _, fn in demos:
            fn()
    elif choice in [d[0] for d in demos]:
        for num, _, fn in demos:
            if num == choice:
                fn()
                break
    else:
        print("  Invalid choice. Exiting.")

    print(f"\n{'═' * 70}")
    print("  Done! All ensemble demonstrations complete.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
