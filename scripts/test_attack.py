import asyncio
from core.config import setup_api_key
from agents.guards_agent import create_red_agent_advance
from core.utils import chat_with_agent

async def main():
    setup_api_key()
    agent, runner = create_red_agent_advance()
    while True:
        prompt = input("\nPrompt: ")
        if not prompt: break
        response, _ = await chat_with_agent(agent, runner, prompt)
        print("Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
