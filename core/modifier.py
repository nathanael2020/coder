import traceback
from openai import OpenAI
from dotenv import load_dotenv
from utils.sanitizer import clean_code
import os
from utils.config import logger

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_API_KEY"),
    base_url="https://api.openai.com/v1",
)


def modify_code(original_code, modification_request):
    """Modify existing code using LLM."""
    logger.info("Starting code modification")
    
    # modification_request = input("\nWhat modifications would you like to make? ")
    # logger.info(f"Modification request: {modification_request}")
    
    system_prompt = """You are a Python code modification expert. Modify the given code according to the user's request.
    Respond with the modified code only, wrapped in ```python and ```. Do not include any other text than the code."""

    prompt = f"""Original code:\n{original_code}\n\nRequested modifications:\n{modification_request}

    Respond with the modified code only, wrapped in ```python and ```. Do not include any other text than the code."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Python code modification expert. Modify the given code according to the user's request."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000
        )
        
        modified_code = clean_code(response.choices[0].message.content)
        logger.info("Code modification successful")
        return modified_code
        
    except Exception as e:
        logger.error(f"Error modifying code: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")

        print(f"Error modifying code: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")

        raise
