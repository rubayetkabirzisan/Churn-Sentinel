# src/test_groq.py
# Verify Groq API connection
# Run: python src/test_groq.py

import os
from dotenv import load_dotenv
load_dotenv()

print("="*55)
print("   CHURN SENTINEL — Groq API Connection Test")
print("="*55)


# ── Test 1: API Key exists ───────────────────────────────
print("\n🔧 Test 1: Checking API key...")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not found in .env file")
    print("   → Add GROQ_API_KEY=gsk_xxxx to your .env file")
    exit(1)
if not api_key.startswith("gsk_"):
    print("❌ API key format wrong — should start with gsk_")
    exit(1)
print(f"✅ API key found → {api_key[:8]}{'*'*20}")


# ── Test 2: Raw Groq connection ──────────────────────────
print("\n🔧 Test 2: Connecting to Groq...")
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [{"role": "user",
                     "content": "Say: Groq connection successful"}],
        max_tokens = 20
    )
    print(f"✅ Groq connected!")
    print(f"   Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Groq connection failed: {e}")
    exit(1)


# ── Test 3: LangChain + Groq ─────────────────────────────
print("\n🔧 Test 3: LangChain → Groq integration...")
try:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        api_key     = api_key,
        model       = "llama-3.1-8b-instant",
        temperature = 0.7,
        max_tokens  = 100
    )
    result = llm.invoke(
        "Write one sentence: why do SaaS customers churn?"
    )
    print(f"✅ LangChain → Groq working!")
    print(f"   Response: {result.content}")
except Exception as e:
    print(f"❌ LangChain error: {e}")
    exit(1)


# ── Test 4: Retention email generation ──────────────────
print("\n🔧 Test 4: Retention email generation...")
try:
    email_prompt = """You are a customer success manager at a SaaS company.
Write a short 3-sentence retention email for a customer who:
- Has been with us for only 2 months
- Is on a month-to-month plan  
- Has high monthly charges of $85/month

Write subject line first, then email body. Be warm and specific."""

    result = llm.invoke(email_prompt)
    print(f"✅ Email generation working!\n")
    print("--- SAMPLE OUTPUT ---")
    print(result.content[:500])
    print("--- END SAMPLE ---")
except Exception as e:
    print(f"❌ Email test failed: {e}")
    exit(1)


# ── Summary ──────────────────────────────────────────────
print("\n" + "="*55)
print("✅ ALL TESTS PASSED")
print("   Groq API + LangChain → ready for agents")
print(f"   Model : llama-3.1-8b-instant")
print(f"   RAM   : 0MB (runs on Groq servers)")
print(f"   Cost  : $0.00")
print("="*55)
