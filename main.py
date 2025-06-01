# Import necessary libraries
import os
from dotenv import load_dotenv  # For loading environment variables from .env file
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel  # AI agent framework
from agents.run import RunConfig  # Configuration for running agents
import asyncio  # For asynchronous programming

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key from environment variables
# This key is needed to authenticate with Google's Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key exists, raise error if not found
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure to define it in .env")

# Create an external OpenAI-compatible client that connects to Google's Gemini API
# This allows us to use Gemini models through OpenAI's interface
external_client = AsyncOpenAI(
    api_key=gemini_api_key,  # Use Gemini API key for authentication
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini's OpenAI-compatible endpoint
)

# Configure the specific model to use (Gemini 2.0 Flash)
# This wraps the Gemini model to work with the agents framework
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",  # Specify which Gemini model variant to use
    openai_client=external_client  # Use our configured Gemini client
)

# Create configuration for running the agent
config = RunConfig(
    model=model,  # The AI model to use for responses
    model_provider=external_client,  # The client that provides access to the model
    tracing_disabled=True  # Disable tracing/logging for cleaner output
)

# Main asynchronous function that runs our AI agent
async def main():
    # Create an AI agent with specific characteristics
    agent = Agent(
        name="Assistant",  # Give the agent a name
        instructions="You are coding Assistant.",  # Define the agent's role and behavior
        model=model  # Specify which model the agent should use
    )
    
    # Run the agent with a specific question and our configuration
    # This sends the question to Gemini and waits for a response
    result = await Runner.run(
        agent, 
        "Tell me about functions in programming.",  # The question/prompt to ask
        run_config=config  # Use our predefined configuration
    )
    
    # Print the agent's response
    print(result.final_output)

# Entry point: run the main function when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())  # Run the asynchronous main function