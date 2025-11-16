"""Simple tests for model registry without pytest dependency."""

import asyncio
from app.core.model_registry import ModelRegistry


async def test_family_inference():
    """Test that model family is correctly inferred."""
    registry = ModelRegistry()

    test_cases = [
        ("mlx-community/gemma-3-4b-it-4bit", "gemma"),
        ("mlx-community/Meta-Llama-3.1-8B-Instruct", "llama"),
        ("mlx-community/Qwen2.5-7B-Instruct", "qwen"),
        ("mlx-community/Phi-3-mini-4k-instruct", "phi"),
        ("mlx-community/Mistral-7B-Instruct-v0.2", "mistral"),
    ]

    for model_id, expected_family in test_cases:
        inferred_family = registry._infer_model_family(model_id)
        assert inferred_family == expected_family, f"Failed for {model_id}: expected {expected_family}, got {inferred_family}"
        print(f"✓ Family inference: {model_id} -> {expected_family}")


async def test_tags_generation():
    """Test that appropriate tags are generated."""
    registry = ModelRegistry()

    # Test LM model with quantization
    tags = registry._generate_model_tags("lm", "mlx-community/gemma-3-4b-it-4bit")
    assert "local" in tags
    assert "chat" in tags
    assert "quantized" in tags
    print(f"✓ Tags for LM model: {tags}")

    # Test embeddings model
    tags = registry._generate_model_tags("embeddings", "mlx-community/bge-base-en-v1.5")
    assert "local" in tags
    assert "embeddings" in tags
    print(f"✓ Tags for embeddings model: {tags}")

    # Test image generation model
    tags = registry._generate_model_tags("image-generation", "flux-schnell")
    assert "local" in tags
    assert "image" in tags
    print(f"✓ Tags for image model: {tags}")


async def test_description_generation():
    """Test that model descriptions are generated correctly."""
    registry = ModelRegistry()

    description = registry._generate_model_description(
        "mlx-community/gemma-3-4b-it-4bit",
        "lm",
        "gemma"
    )
    assert "Gemma" in description
    assert "language model" in description
    assert "MLX" in description
    print(f"✓ Description: {description}")


async def test_model_registration():
    """Test model registration with rich metadata."""
    registry = ModelRegistry()

    # Mock handler
    class MockHandler:
        pass

    handler = MockHandler()

    # Register a model
    await registry.register_model(
        model_id="mlx-community/gemma-3-4b-it-4bit",
        handler=handler,
        model_type="lm",
        context_length=8192,
    )

    # Check it's registered
    assert registry.has_model("mlx-community/gemma-3-4b-it-4bit")
    print("✓ Model registered successfully")

    # Get metadata
    metadata = registry.get_metadata("mlx-community/gemma-3-4b-it-4bit")
    assert metadata.id == "mlx-community/gemma-3-4b-it-4bit"
    assert metadata.type == "lm"
    assert metadata.context_length == 8192
    assert metadata.family == "gemma"
    assert "local" in metadata.tags
    assert "chat" in metadata.tags
    assert metadata.tier == "3A"
    print(f"✓ Metadata: family={metadata.family}, tags={metadata.tags}, tier={metadata.tier}")

    # List models
    models = registry.list_models()
    assert len(models) == 1
    model = models[0]
    assert model["id"] == "mlx-community/gemma-3-4b-it-4bit"
    assert model["context_length"] == 8192
    assert model["family"] == "gemma"
    assert "tags" in model
    assert model["tier"] == "3A"
    print(f"✓ List models returns rich metadata")


async def main():
    """Run all tests."""
    print("Running model registry tests...\n")

    try:
        await test_family_inference()
        print()

        await test_tags_generation()
        print()

        await test_description_generation()
        print()

        await test_model_registration()
        print()

        print("✅ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
