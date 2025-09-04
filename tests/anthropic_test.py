# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Anthropic provider."""

from unittest import mock
from absl.testing import absltest

from langextract import exceptions
from langextract.core import data
from langextract.providers import anthropic


class AnthropicProviderTest(absltest.TestCase):
  """Test Anthropic provider functionality."""

  def test_provider_init_requires_api_key(self):
    """Anthropic provider should require an API key."""
    with self.assertRaises(exceptions.InferenceConfigError) as cm:
      anthropic.AnthropicLanguageModel()

    self.assertIn("API key not provided", str(cm.exception))

  def test_provider_init_with_api_key(self):
    """Anthropic provider should initialize with API key."""
    provider = anthropic.AnthropicLanguageModel(api_key="test-key")
    
    self.assertEqual(provider.api_key, "test-key")
    self.assertEqual(provider.model_id, "claude-3-5-sonnet-20241022")
    self.assertEqual(provider.format_type, data.FormatType.JSON)

  def test_provider_init_with_custom_parameters(self):
    """Anthropic provider should accept custom parameters."""
    provider = anthropic.AnthropicLanguageModel(
        model_id="claude-3-haiku-20240307",
        api_key="test-key",
        base_url="https://api.anthropic.com",
        format_type=data.FormatType.YAML,
        temperature=0.7,
        max_workers=5,
    )
    
    self.assertEqual(provider.model_id, "claude-3-haiku-20240307")
    self.assertEqual(provider.api_key, "test-key")
    self.assertEqual(provider.base_url, "https://api.anthropic.com")
    self.assertEqual(provider.format_type, data.FormatType.YAML)
    self.assertEqual(provider.temperature, 0.7)
    self.assertEqual(provider.max_workers, 5)

  def test_requires_fence_output_json_false(self):
    """JSON format should not require fence output."""
    provider = anthropic.AnthropicLanguageModel(
        api_key="test-key", format_type=data.FormatType.JSON
    )
    self.assertFalse(provider.requires_fence_output)

  def test_requires_fence_output_yaml_true(self):
    """YAML format should require fence output."""
    provider = anthropic.AnthropicLanguageModel(
        api_key="test-key", format_type=data.FormatType.YAML
    )
    self.assertTrue(provider.requires_fence_output)

  @mock.patch("anthropic.Anthropic")
  def test_inference_call_structure(self, mock_anthropic_class):
    """Test that inference calls Anthropic API with correct structure."""
    # Setup mock
    mock_client = mock.MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = mock.MagicMock()
    mock_response.content = [mock.MagicMock()]
    mock_response.content[0].text = '{"result": "test"}'
    mock_client.messages.create.return_value = mock_response

    # Create provider and run inference
    provider = anthropic.AnthropicLanguageModel(api_key="test-key")
    results = list(provider.infer(["Test prompt"]))

    # Verify client creation
    mock_anthropic_class.assert_called_once_with(api_key="test-key")

    # Verify API call
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args[1]
    
    self.assertEqual(call_args["model"], "claude-3-5-sonnet-20241022")
    self.assertEqual(call_args["messages"], [{"role": "user", "content": "Test prompt"}])
    self.assertEqual(call_args["system"], "You are a helpful assistant that responds in JSON format.")
    self.assertEqual(call_args["max_tokens"], 4096)
    
    # Verify results
    self.assertEqual(len(results), 1)
    self.assertEqual(len(results[0]), 1)
    self.assertEqual(results[0][0].output, '{"result": "test"}')
    self.assertEqual(results[0][0].score, 1.0)

  @mock.patch("anthropic.Anthropic")
  def test_inference_with_parameters(self, mock_anthropic_class):
    """Test that inference passes parameters correctly."""
    # Setup mock
    mock_client = mock.MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = mock.MagicMock()
    mock_response.content = [mock.MagicMock()]
    mock_response.content[0].text = "response"
    mock_client.messages.create.return_value = mock_response

    # Create provider and run inference with parameters
    provider = anthropic.AnthropicLanguageModel(
        api_key="test-key", temperature=0.5
    )
    results = list(provider.infer(
        ["Test prompt"], 
        temperature=0.8,
        max_output_tokens=1000,
        top_p=0.9,
        top_k=40,
        stop_sequences=["END"]
    ))

    # Verify API call includes all parameters
    call_args = mock_client.messages.create.call_args[1]
    self.assertEqual(call_args["temperature"], 0.8)  # Should override provider temperature
    self.assertEqual(call_args["max_tokens"], 1000)
    self.assertEqual(call_args["top_p"], 0.9)
    self.assertEqual(call_args["top_k"], 40)
    self.assertEqual(call_args["stop_sequences"], ["END"])

  @mock.patch("anthropic.Anthropic")
  def test_yaml_format_system_message(self, mock_anthropic_class):
    """Test that YAML format uses appropriate system message."""
    # Setup mock
    mock_client = mock.MagicMock()
    mock_anthropic_class.return_value = mock_client
    
    mock_response = mock.MagicMock()
    mock_response.content = [mock.MagicMock()]
    mock_response.content[0].text = "response"
    mock_client.messages.create.return_value = mock_response

    # Create provider with YAML format
    provider = anthropic.AnthropicLanguageModel(
        api_key="test-key", format_type=data.FormatType.YAML
    )
    list(provider.infer(["Test prompt"]))

    # Verify YAML system message
    call_args = mock_client.messages.create.call_args[1]
    self.assertEqual(call_args["system"], "You are a helpful assistant that responds in YAML format.")

  def test_provider_init_without_anthropic_package(self):
    """Test error when anthropic package is not available."""
    with mock.patch.dict("sys.modules", {"anthropic": None}):
      with self.assertRaises(exceptions.InferenceConfigError) as cm:
        anthropic.AnthropicLanguageModel(api_key="test-key")

      self.assertIn("anthropic package", str(cm.exception))
      self.assertIn("pip install langextract[anthropic]", str(cm.exception))


if __name__ == "__main__":
  absltest.main()