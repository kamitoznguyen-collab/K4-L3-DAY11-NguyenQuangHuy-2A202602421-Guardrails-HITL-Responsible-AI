import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
from core.config import setup_api_key
from attacks.attacks import generate_ai_attacks, run_attacks
from agents.guards_agent import create_red_agent_advance

async def main():
    setup_api_key()
    print("Generating AI attacks...")
    prompts = await generate_ai_attacks()
    if not prompts:
        print("Failed to generate AI attacks.")
        return
    
    agent, runner = create_red_agent_advance()
    print("Running AI generated attacks on guards agent...")
    await run_attacks(agent, runner, prompts, target_name="guards", save_json=False)

if __name__ == "__main__":
    asyncio.run(main())
