import requests
import json
import anthropic
import os

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])


def generate_with_ollama(prompt, model="mistral:7b-instruct-q4_K_M", stream=True):
    """Generate a response using Ollama with streaming support."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    response = requests.post(url, json=payload, stream=True)
    if response.status_code == 200:
        if stream:
            # Stream the response
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        yield chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
        else:
            # Return the full response
            return response.json().get("response", "")
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    
    
    
def generate_with_claude(messages, model="claude-3-5-sonnet-20241022", max_tokens=1000):
    """Generate a response using Claude."""
    try:
        response = client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.5,  # Adjust for creativity
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Error generating response with Claude: {e}")