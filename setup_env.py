"""
First, create a .env file in the root directory of the project that contains your API keys, for example:
    OPENAI_API_KEY=your-openai-api-key-here
Then run this to set up environment variables with API keys.
"""

from pathlib import Path

def load_env():    
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        print("❌ No .env file found")
        return False
    
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    print("✅ Environment configured correctly")
    return True


if __name__ == "__main__":
    load_env()
