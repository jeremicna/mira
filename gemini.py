import os
import sys
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY is not set in environment or .env", file=sys.stderr)
    sys.exit(1)

genai.configure(api_key=API_KEY)

# --- SYSTEM PROMPT ---
system_prompt = """You are a helpful assistant that analyzes conversation transcripts.
Your goal is to extract key personal details, summarize the main topics discussed, and generate intelligent follow-up items for the next conversation.
You will be given a user prompt containing the conversation, which is a list of 'segments' with 'speaker' and 'text'.

The user will now send MULTIPLE transcripts (an array of transcript objects), and you must analyze *all* of them together as one continuous inferred relationship.

Analyze all transcripts to:
1. **Identify Key Info:** Extract the name, occupation, and relationship of the person the user was talking to. Infer if possible or use "Unknown".
2. **Summarize Key Points:** Identify the 3–5 most important recurring topics or important events across all transcripts.
3. **Generate Action Items:** Create 2–3 brief follow-up questions or topics for the *next* call.

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

def get_json_analysis(conversation_list: list) -> str:
    """
    Accepts a LIST of transcripts, not a single dict.
    Each item in conversation_list is expected to be:
       { "segments": [ {speaker, text}, ... ] }

    Returns a JSON string response from Gemini.
    """

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )
    except Exception as e:
        print(f"Error creating model (check library version): {e}", file=sys.stderr)
        return None

    # Create the user prompt with the ENTIRE array of transcripts
    user_prompt = (
        "Here are multiple conversation transcripts. "
        "Analyze them all together as one combined relationship/context: "
        f"{json.dumps(conversation_list)}"
    )

    generation_config = {"response_mime_type": "application/json"}

    print("Sending request to Gemini API...", file=sys.stderr)

    try:
        response = model.generate_content(
            user_prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        return None


# --- Standalone test block ---
if __name__ == "__main__":

    print("--- Running gemini.py in standalone test mode ---", file=sys.stderr)

    # TWO transcripts now
    test_conversation_data = [
        {
            "segments": [
                {"speaker": "SPEAKER_01", "text": "Hey Alice, how was the doctor visit?"},
                {"speaker": "SPEAKER_02", "text": "It went fine, they just adjusted my medication."},
                {"speaker": "SPEAKER_01", "text": "How's work been?"},
                {"speaker": "SPEAKER_02", "text": "Stressful! These kids in my class are exhausting."}
            ]
        },
        {
            "segments": [
                {"speaker": "SPEAKER_01", "text": "Hey Alice, did you ever sort out that issue with your car?"},
                {"speaker": "SPEAKER_02", "text": "Oh, finally! The mechanic said it was just the battery, thankfully not something expensive."},
                {"speaker": "SPEAKER_01", "text": "That's a relief. Are you still planning that short trip you mentioned?"},
                {"speaker": "SPEAKER_02", "text": "Yeah, I’m hoping to go next weekend if work doesn’t blow up again."},
                {"speaker": "SPEAKER_01", "text": "You deserve a break. Let me know if you need help watching the kids before you leave."},
                {"speaker": "SPEAKER_02", "text": "Thanks, that means a lot. I’ll let you know once I finalize plans."}
            ]
        }
    ]

    json_output_string = get_json_analysis(test_conversation_data)

    if json_output_string:
        print("\nGemini response (JSON string):\n")
        print(json_output_string)
    else:
        print("Failed to get a response from Gemini.", file=sys.stderr)
