import traceback
from openai import OpenAI
import os
from utils.sanitizer import clean_code
from utils.config import client, logger

def generate_code(user_request):
    """Generate Python code from natural language request."""
    system_prompt = """You are an expert Python programmer. Generate clean, efficient, and safe code.
    Include proper error handling and input validation. Only respond with executable code wrapped in ```python and ```.
    
    Never include any other text than the code."""

    prompt = f"Write Python code to {user_request}. Respond with code only, wrapped in ```python and ```. Do not include any other text than the code."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to the latest model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.7,
        )

        print(f"Generated code: {response.choices[0].message.content}")
        logger.info(f"Generated code: {response.choices[0].message.content}")

        return clean_code(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Code generation failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        print(f"Code generation failed: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(f"Code generation failed: {str(e)}")
