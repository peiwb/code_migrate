"""
LLM Service Test Module
Tests the integration between CortexLLMService and Snowflake Cortex API

This test module implements all test cases defined in the architecture design document,
validating the integration functionality of CortexLLMService with external Snowflake Cortex API.
"""

import pytest
from services.llm_service import CortexLLMService


@pytest.fixture(scope="module")
def llm_service():
    """
    Test fixture for creating CortexLLMService instance
    Uses module scope to ensure only one instance is created for the entire test module
    """
    service = CortexLLMService()
    return service


def test_service_initialization(llm_service):
    """
    Test Case 1: Verify that CortexLLMService can be successfully instantiated

    Purpose: Validate service initialization and Snowflake Session creation
    """
    # Verify service instance is not None
    assert llm_service is not None

    # Verify service has session attribute
    assert hasattr(llm_service, 'session')

    # Verify session is not None
    assert llm_service.session is not None


def test_get_text_completion_success(llm_service):
    """
    Test Case 2: Verify the success path of get_text_completion method

    Purpose: Validate successful text response retrieval
    """
    # Define simple and clear prompt
    prompt = "Say 'Hello, World!'"

    # Call text completion method
    response = llm_service.get_text_completion(prompt)

    # Verify response is string type
    assert isinstance(response, str)

    # Verify response is not empty
    assert len(response) > 0

    # Verify response contains expected content (case insensitive)
    assert "hello" in response.lower() and "world" in response.lower()


def test_get_json_completion_success(llm_service):
    """
    Test Case 3: Verify the success path of get_json_completion method

    Purpose: Validate successful structured JSON response retrieval
    """
    # Define prompt requiring structured output
    prompt = "Extract the user's name and age from the following sentence: The user, named Alex, is 35 years old."

    # Define JSON Schema
    test_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        },
        "required": ["name", "age"]
    }

    # Call JSON completion method
    response = llm_service.get_json_completion(prompt, json_schema=test_schema)

    # Verify response is dictionary type
    assert isinstance(response, dict)

    # Verify response contains required fields
    assert "name" in response
    assert "age" in response

    # Verify field types
    assert isinstance(response["name"], str)
    assert isinstance(response["age"], (int, float))

    # Verify extracted content accuracy
    assert response["name"].lower() == "alex"
    assert response["age"] == 35


def test_get_text_completion_complex_prompt(llm_service):
    """
    Test Case 4: Test text completion with complex prompt

    Purpose: Verify service can handle more complex text generation tasks
    """
    prompt = "Write a brief summary about artificial intelligence in exactly 2 sentences."

    response = llm_service.get_text_completion(prompt)

    # Verify basic response format
    assert isinstance(response, str)
    assert len(response) > 0

    # Verify response contains relevant keywords
    response_lower = response.lower()
    assert any(keyword in response_lower for keyword in ["artificial", "intelligence", "ai", "machine", "learning"])


def test_get_json_completion_complex_schema(llm_service):
    """
    Test Case 5: Test structured completion with complex JSON Schema

    Purpose: Verify service can handle more complex structured output
    """
    prompt = """
    Analyze the following text and extract information:
    "John Smith, 28, works as a Software Engineer at TechCorp.
    He has 5 years of experience and his email is john.smith@techcorp.com"
    """

    complex_schema = {
        "type": "object",
        "properties": {
            "personal_info": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"}
                },
                "required": ["name", "age"]
            },
            "professional_info": {
                "type": "object",
                "properties": {
                    "job_title": {"type": "string"},
                    "company": {"type": "string"},
                    "experience_years": {"type": "number"},
                    "email": {"type": "string"}
                },
                "required": ["job_title", "company", "experience_years", "email"]
            }
        },
        "required": ["personal_info", "professional_info"]
    }

    response = llm_service.get_json_completion(prompt, json_schema=complex_schema)

    # Verify top-level structure
    assert isinstance(response, dict)
    assert "personal_info" in response
    assert "professional_info" in response

    # Verify personal info structure
    personal = response["personal_info"]
    assert isinstance(personal, dict)
    assert "name" in personal
    assert "age" in personal
    assert isinstance(personal["name"], str)
    assert isinstance(personal["age"], (int, float))

    # Verify professional info structure
    professional = response["professional_info"]
    assert isinstance(professional, dict)
    assert "job_title" in professional
    assert "company" in professional
    assert "experience_years" in professional
    assert "email" in professional

    # Verify data types
    assert isinstance(professional["job_title"], str)
    assert isinstance(professional["company"], str)
    assert isinstance(professional["experience_years"], (int, float))
    assert isinstance(professional["email"], str)


def test_service_session_persistence(llm_service):
    """
    Test Case 6: Verify service session persistence

    Purpose: Ensure Snowflake Session remains valid across multiple calls
    """
    # First call
    prompt1 = "What is 2+2?"
    response1 = llm_service.get_text_completion(prompt1)

    # Verify first call success
    assert isinstance(response1, str)
    assert len(response1) > 0

    # Second call
    prompt2 = "What is the capital of France?"
    response2 = llm_service.get_text_completion(prompt2)

    # Verify second call success
    assert isinstance(response2, str)
    assert len(response2) > 0

    # Verify both calls use the same session
    assert llm_service.session is not None

    # Verify responses are different (ensure not cached responses)
    assert response1 != response2


if __name__ == "__main__":
    # If running this file directly, execute tests
    pytest.main([__file__, "-v"])