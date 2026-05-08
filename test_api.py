import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Test 1 - Vague
        print("Test 1: Vague query")
        req1 = {"messages": [{"role": "user", "content": "I need an assessment"}]}
        res1 = await client.post("/chat", json=req1, timeout=35.0)
        print(res1.status_code, res1.json())
        print("-" * 40)

        # Test 2 - Enough context
        print("Test 2: Enough context")
        req2 = {"messages": [
            {"role": "user", "content": "Hiring a mid-level Java developer who collaborates with stakeholders"},
            {"role": "assistant", "content": "Got it. Do you need a cognitive test or personality test?"},
            {"role": "user", "content": "Around 4 years experience, need to test both technical skills and communication"}
        ]}
        res2 = await client.post("/chat", json=req2, timeout=35.0)
        print(res2.status_code, res2.json())
        print("-" * 40)

        # Test 5 - Out of scope
        print("Test 5: Out of scope")
        req5 = {"messages": [{"role": "user", "content": "What are the legal requirements for pre-employment testing in the UK?"}]}
        res5 = await client.post("/chat", json=req5, timeout=35.0)
        print(res5.status_code, res5.json())
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
