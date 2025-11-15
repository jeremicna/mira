import os
import sys
from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY is not set in environment or .env", file=sys.stderr)
    sys.exit(1)

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

def call_gemini(prompt: str) -> str:
    """
    Send a simple text prompt to Gemini and return the model's text response.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    # Handle empty responses defensively
    try:
        return response.text
    except Exception:
        return str(response)

if __name__ == "__main__":
    # Simple example prompt
    prompt = "Explain what an API is in one short paragraph."
    output = call_gemini(prompt)
    print("Gemini response:\n")
    print(output)