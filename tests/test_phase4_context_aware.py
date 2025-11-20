"""
Test script for Phase-4 context-aware completions.
Tests RAG context injection and HTDI entity context.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_context_aware_completion_with_rag():
    """Test chat completion with RAG context chunks."""
    print("\n=== Test 1: RAG Context Injection ===")

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer questions accurately."
            },
            {
                "role": "user",
                "content": "What is the temperature in the peace room?"
            }
        ],
        "context": [
            {
                "text": "The peace room is located on the second floor of the building. It has a temperature sensor that monitors ambient temperature.",
                "score": 0.92,
                "metadata": {
                    "source": "rag",
                    "collection": "rooms"
                }
            },
            {
                "text": "The peace room maintains a comfortable temperature between 20-24°C for optimal studying conditions.",
                "score": 0.87,
                "metadata": {
                    "source": "rag",
                    "collection": "facilities"
                }
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Success!")
            print(f"Response: {result['choices'][0]['message']['content']}")

            # Check for context metadata
            if 'htdi' in result:
                print(f"\nContext Metadata:")
                print(f"  - Context Used: {result['htdi']['context_used']}")
                print(f"  - Context Sources: {result['htdi']['context_sources']}")
                print(f"  - Context Count: {result['htdi']['context_count']}")
            else:
                print("\n⚠ Warning: No context metadata in response")
        else:
            print(f"✗ Failed: {response.text}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")


def test_context_aware_completion_with_htdi():
    """Test chat completion with HTDI entity context."""
    print("\n=== Test 2: HTDI Entity Context Injection ===")

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": "What's the current temperature and lighting status in the room?"
            }
        ],
        "htdi": {
            "roomId": "peace",
            "entities": [
                {
                    "entityId": "sensor.peace_temperature",
                    "state": "22.5",
                    "attributes": {
                        "unit_of_measurement": "°C"
                    }
                },
                {
                    "entityId": "light.peace_main",
                    "state": "on",
                    "attributes": {
                        "brightness": 80
                    }
                }
            ]
        },
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Success!")
            print(f"Response: {result['choices'][0]['message']['content']}")

            # Check for context metadata
            if 'htdi' in result:
                print(f"\nContext Metadata:")
                print(f"  - Context Used: {result['htdi']['context_used']}")
                print(f"  - Context Sources: {result['htdi']['context_sources']}")
                print(f"  - Context Count: {result['htdi']['context_count']}")
            else:
                print("\n⚠ Warning: No context metadata in response")
        else:
            print(f"✗ Failed: {response.text}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")


def test_context_aware_completion_with_both():
    """Test chat completion with both RAG and HTDI context."""
    print("\n=== Test 3: Combined RAG + HTDI Context ===")

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "You are a smart building assistant."
            },
            {
                "role": "user",
                "content": "Is the peace room comfortable for studying right now?"
            }
        ],
        "context": [
            {
                "text": "The peace room is designed for quiet study and concentration. Optimal conditions are 21-23°C and moderate lighting.",
                "score": 0.89,
                "metadata": {
                    "source": "rag",
                    "collection": "room_guidelines"
                }
            }
        ],
        "htdi": {
            "roomId": "peace",
            "entities": [
                {
                    "entityId": "sensor.peace_temperature",
                    "state": "22.0",
                    "attributes": {
                        "unit_of_measurement": "°C"
                    }
                },
                {
                    "entityId": "sensor.peace_noise_level",
                    "state": "35",
                    "attributes": {
                        "unit_of_measurement": "dB"
                    }
                }
            ]
        },
        "max_tokens": 150,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Success!")
            print(f"Response: {result['choices'][0]['message']['content']}")

            # Check for context metadata
            if 'htdi' in result:
                print(f"\nContext Metadata:")
                print(f"  - Context Used: {result['htdi']['context_used']}")
                print(f"  - Context Sources: {result['htdi']['context_sources']}")
                print(f"  - Context Count: {result['htdi']['context_count']}")
            else:
                print("\n⚠ Warning: No context metadata in response")
        else:
            print(f"✗ Failed: {response.text}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")


def test_without_context():
    """Test normal chat completion without context (baseline)."""
    print("\n=== Test 4: Baseline (No Context) ===")

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ Success!")
            print(f"Response: {result['choices'][0]['message']['content']}")

            # Check that no context metadata is present
            if 'htdi' in result and result['htdi']:
                print(f"\n⚠ Warning: Unexpected context metadata in response")
            else:
                print("\n✓ No context metadata (as expected)")
        else:
            print(f"✗ Failed: {response.text}")

    except Exception as e:
        print(f"✗ Error: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase-4 Context-Aware Completion Tests")
    print("=" * 60)
    print("\nNOTE: These tests require the MLX OpenAI server to be running.")
    print("Start the server with: python -m app.main --model-path <model> --model-type lm")
    print()

    # Run tests
    test_context_aware_completion_with_rag()
    test_context_aware_completion_with_htdi()
    test_context_aware_completion_with_both()
    test_without_context()

    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)
