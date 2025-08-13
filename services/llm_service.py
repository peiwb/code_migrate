"""LLM Service Layer (llm_service.py) - V1.0

This module is the sole component responsible for communicating with Large Language Models (LLM)
throughout the entire project. It encapsulates all complexities related to underlying LLM APIs
and provides a simple, clean, and unified interface to external consumers.
"""

import json
import os
from dotenv import load_dotenv
import anthropic


class LLMServiceError(Exception):
    """Custom exception class for LLM service"""
    pass


class CortexLLMService:
    """
    Claude 4 API LLM Service Class (Compatible with Cortex interface)

    Responsible for managing connection with Claude 4 API and providing
    unified LLM calling interfaces. Agents only need to focus on "what questions to ask"
    without worrying about how to call the LLM.
    """

    def __init__(self):
        """
        Initialize LLM service instance and establish connection with Claude 4 API.
        This method should only be called once during the program lifecycle.

        Raises:
            LLMServiceError: Raised when API client initialization fails
        """
        try:
            # Load environment variables from .env file
            load_dotenv()

            # Get API key from environment variable
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

            # Initialize Claude 4 API client
            self.client = anthropic.Anthropic(api_key=api_key)

            # Create a session-like object for backward compatibility
            # This allows other modules to continue using llm_service.session
            self.session = self._create_session_object()

        except Exception as e:
            raise LLMServiceError(f"Failed to initialize Claude 4 API client: {e}")

    def _create_session_object(self):
        """
        Create a session-like object for backward compatibility.
        This allows other modules to access session attributes without breaking.
        """
        class SessionObject:
            def __init__(self, client):
                self.client = client
                # Add any session attributes that other modules might expect
                self.connected = True
                self.api_key = self.client.api_key

            def __getattr__(self, name):
                # Forward any unknown attributes to the client
                return getattr(self.client, name)

            def __repr__(self):
                return f"<Claude4Session connected={self.connected}>"

        return SessionObject(self.client)

    def get_text_completion(self, prompt: str) -> str:
        """
        Send a prompt and get a plain text response.

        Args:
            prompt (str): Complete prompt text to send to the LLM

        Returns:
            str: Plain text string returned by the LLM

        Raises:
            LLMServiceError: Raised when API call fails
        """
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.content[0].text

        except Exception as e:
            raise LLMServiceError(f"Failed to get text completion from Claude 4 API: {e}")

    def get_json_completion(self, prompt: str, json_schema: dict) -> dict:
        """
        Send a prompt and force the LLM to return a JSON object conforming to specified structure.

        **IMPORTANT**: This method uses positional parameters to match Cortex API interface:
        - First parameter: prompt (str)
        - Second parameter: json_schema (dict)

        Args:
            prompt (str): Complete prompt text to send to the LLM
            json_schema (dict): Python dictionary describing expected JSON structure and types,
                               following JSON Schema specification

        Returns:
            dict: Structured response parsed from JSON string to Python dictionary

        Raises:
            LLMServiceError: Raised when API call fails or JSON parsing fails
        """
        try:
            # Build enhanced prompt that explicitly requests JSON format
            enhanced_prompt = f"""{prompt}

Please respond with a JSON object that strictly follows this schema:
{json.dumps(json_schema, indent=2)}

Your response must be valid JSON that can be parsed directly."""

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ]
            )

            # Extract the response text
            response_text = response.content[0].text.strip()

            # Try to parse as JSON
            try:
                parsed_response = json.loads(response_text)
                return parsed_response
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                # Look for JSON block between ```json and ``` or just {}
                import re
                json_pattern = r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*\})'
                matches = re.findall(json_pattern, response_text, re.DOTALL)

                for match in matches:
                    for group in match:
                        if group.strip():
                            try:
                                parsed_response = json.loads(group.strip())
                                return parsed_response
                            except json.JSONDecodeError:
                                continue

                # If all parsing attempts fail, raise error
                raise json.JSONDecodeError("Could not extract valid JSON from response", response_text, 0)

        except json.JSONDecodeError as e:
            raise LLMServiceError(f"Failed to parse JSON response from Claude 4 API: {e}")
        except Exception as e:
            raise LLMServiceError(f"Failed to get JSON completion from Claude 4 API: {e}")