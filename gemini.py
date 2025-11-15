import os
import sys
import json
from dotenv import load_dotenv
import google.generativeai as genai  # <-- 1. This is the correct import

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY is not set in environment or .env", file=sys.stderr)
    sys.exit(1)

# --- 2. Configure the API key this way ---
genai.configure(api_key=API_KEY)

# --- SYSTEM PROMPT (Instructions) ---
system_prompt = """You are a helpful assistant that analyzes conversation transcripts.
Your goal is to extract key personal details, summarize the main topics discussed, and generate intelligent follow-up items for the next conversation.
You will be given a user prompt containing the conversation, which is a list of 'segments' with 'speaker' and 'text'.
Analyze the conversation to:
1. **Identify Key Info:** Extract the name, occupation, and relationship of the person the user was talking to. Infer if possible or use "Unknown".
2. **Summarize Key Points:** Identify the 3-5 most important topics, facts, or events.
3. **Generate Action Items:** Create 2-3 brief follow-up questions or topics for the *next* call.
You MUST respond with ONLY a single, valid JSON object. Do not include any other text, markdown formatting, or explanations.
Your output MUST strictly follow this exact JSON structure:
{
     "name": "Alice Clark",
     "occupation":  "Teacher",
     "relationship": "Daughter",
     "last_points": [
          "You talked about your doctor visit on Monday",
          "Alice explained the new medication schedule", 
          "You asked about her kids and Jake scored goal"
     ], 
     "convo_points": [
          "Ask how her kids are doing this week", 
          "Ask if Sunday still works for the call"
     ]
}"""

def get_json_analysis(conversation_dict: dict) -> str:
    """
    Sends a conversation to Gemini and returns a structured JSON analysis.
    The conversation_dict is *NOT* hardcoded. It is passed in as an argument.
    """
    
    # --- 3. Create the model, passing the system prompt at initialization ---
    # (This is the modern, correct way to do this)
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )
    except Exception as e:
        print(f"Error creating model (check library version): {e}", file=sys.stderr)
        return None

    # --- 4. CREATE THE USER PROMPT (The Data) ---
    user_prompt = f"Here is the conversation transcript: {json.dumps(conversation_dict)}"
    
    # --- 5. FORCE JSON OUTPUT (using a dictionary) ---
    generation_config = {"response_mime_type": "application/json"}

    print("Sending request to Gemini API...", file=sys.stderr)
    
    try:
        # --- 6. MAKE THE CORRECT API CALL ---
        # (Note: no 'system_instruction' here, it's in the model)
        response = model.generate_content(
            user_prompt,  # <-- Pass data here
            generation_config=generation_config # <-- Pass the config
        )
        # The response.text will be a clean JSON string
        return response.text
    except Exception as e:
        # This will now catch any *real* API errors
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        return None

# --- This block is just for testing this file directly ---
if __name__ == "__main__":
    
    print("--- Running gemini.py in standalone test mode ---", file=sys.stderr)

    # This is a MOCK (hardcoded) transcript
    test_conversation_data = {
      "segments": [
        {"speaker": "SPEAKER_01", "text": "Hi Alice, its been a while, how was your work week?"},
        {"speaker": "SPEAKER_02", "text": "Oh, hi! It was fine. I had that doctor visit on Monday, you know."},
        {"speaker": "SPEAKER_01", "text": "Right, what did they say?"},
        {"speaker": "SPEAKER_02", "text": "It's all good, they just explained the new medication schedule."},
        {"speaker": "SPEAKER_02", "text": "Anyways, how's mom, I've really missed the both of you."},
        {"speaker": "SPEAKER_01", "text": "She's alright, we have missed you kids too."},
        {"speaker": "SPEAKER_02", "text": "Yea, sorry for not visiting more, work has been stressful, these kids in my class are hard to handle."},
      ]
    }
    
    # Call the function with the test data
    json_output_string = get_json_analysis(test_conversation_data)
    
    if json_output_string:
        print("\nGemini response (JSON string):\n")
        print(json_output_string)
    else:
        print("Failed to get a response from Gemini.", file=sys.stderr)