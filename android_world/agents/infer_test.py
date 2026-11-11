# Copyright 2026 The android_world Authors.
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

import os
import time
from unittest import mock
from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import llm_usage
from google.genai import models as genai_models
from google.genai import types as genai_types
import requests


def _gemini_response(candidates, usage=None):
  return genai_types.GenerateContentResponse(
      candidates=candidates,
      usage_metadata=genai_types.GenerateContentResponseUsageMetadata(**usage)
      if usage
      else None,
  )


def _text_candidate(text):
  return genai_types.Candidate(
      content=genai_types.Content(parts=[genai_types.Part(text=text)])
  )


class InferTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_post = mock.patch.object(requests, "post").start()
    self.mock_sleep = mock.patch.object(time, "sleep").start()
    self.mock_generate_content = mock.patch.object(
        genai_models.Models, "generate_content"
    ).start()
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    os.environ["GCP_API_KEY"] = "fake_api_key"
    llm_usage.get_tracker().reset()

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_gemini_gcp(self):
    self.mock_generate_content.return_value = _gemini_response(
        [_text_candidate("fake response")]
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, "fake response")
    self.assertEqual(is_safe, True)

  def test_gemini_gcp_passes_seed_and_temperature(self):
    self.mock_generate_content.return_value = _gemini_response(
        [_text_candidate("ok")]
    )
    llm = infer.GeminiGcpWrapper(
        model_name="some_gemini_model",
        temperature=0.0,
        seed=7,
        thinking_level="low",
    )
    llm.predict("fake prompt")
    config = self.mock_generate_content.call_args.kwargs["config"]
    self.assertEqual(config.seed, 7)
    self.assertEqual(config.temperature, 0.0)
    self.assertEqual(
        config.thinking_config.thinking_level, genai_types.ThinkingLevel.LOW
    )
    self.assertEqual(
        self.mock_generate_content.call_args.kwargs["model"],
        "some_gemini_model",
    )

  def test_gemini_gcp_records_usage(self):
    self.mock_generate_content.return_value = _gemini_response(
        [_text_candidate("fake response")],
        usage=dict(
            prompt_token_count=100,
            candidates_token_count=20,
            thoughts_token_count=5,
            cached_content_token_count=40,
            total_token_count=125,
        ),
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    llm.predict_mm("fake prompt", [])
    (call,) = llm_usage.get_tracker().calls
    self.assertEqual(call.model, "some_gemini_model")
    self.assertEqual(call.input_tokens, 100)
    self.assertEqual(call.output_tokens, 20)
    self.assertEqual(call.thinking_tokens, 5)
    self.assertEqual(call.cached_tokens, 40)
    self.assertEqual(call.total_tokens, 125)
    self.assertEqual(call.attempts, 1)
    self.assertTrue(call.success)

  def test_gemini_gcp_retry_sums_usage(self):
    usage = dict(prompt_token_count=10, candidates_token_count=0,
                 total_token_count=10)
    self.mock_generate_content.side_effect = [
        _gemini_response([genai_types.Candidate(content=genai_types.Content(
            parts=[]))], usage=usage),
        _gemini_response([_text_candidate("ok")], usage=usage),
    ]
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, _, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, "ok")
    (call,) = llm_usage.get_tracker().calls
    self.assertEqual(call.attempts, 2)
    self.assertEqual(call.input_tokens, 20)

  def test_gemini_gcp_error(self):
    self.mock_generate_content.return_value = _gemini_response(
        [genai_types.Candidate(content=genai_types.Content(parts=[]))]
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, output = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertIsNone(is_safe)
    self.assertIsNone(output)
    (call,) = llm_usage.get_tracker().calls
    self.assertFalse(call.success)
    self.assertEqual(call.attempts, 3)

  def test_gemini_gcp_no_candidates(self):
    self.mock_generate_content.return_value = _gemini_response([])
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, output = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertIsNone(is_safe)
    self.assertIsNone(output)

  def test_gemini_gcp_unsafe(self):
    self.mock_generate_content.return_value = _gemini_response([
        genai_types.Candidate(
            content=genai_types.Content(parts=[]),
            finish_reason=genai_types.FinishReason.SAFETY,
        )
    ])
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertEqual(is_safe, False)

  def test_gpt4v(self):
    llm = infer.Gpt4Wrapper(model_name="gpt-4-turbo-2024-04-09")
    mock_200_response = requests.Response()
    mock_200_response.status_code = 200
    mock_200_response._content = (
        b'{"choices": [{"message": {"content": "fake response"}}]}'
    )
    self.mock_post.return_value = mock_200_response

    text_output, _, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, "fake response")

  def test_gpt4v_records_usage(self):
    llm = infer.Gpt4Wrapper(model_name="gpt-4-turbo-2024-04-09", seed=3)
    mock_200_response = requests.Response()
    mock_200_response.status_code = 200
    mock_200_response._content = (
        b'{"choices": [{"message": {"content": "fake response"}}],'
        b' "usage": {"prompt_tokens": 50, "completion_tokens": 12,'
        b' "total_tokens": 62,'
        b' "completion_tokens_details": {"reasoning_tokens": 2},'
        b' "prompt_tokens_details": {"cached_tokens": 10}}}'
    )
    self.mock_post.return_value = mock_200_response
    llm.predict_mm("fake prompt", [])
    self.assertEqual(self.mock_post.call_args.kwargs["json"]["seed"], 3)
    (call,) = llm_usage.get_tracker().calls
    self.assertEqual(call.input_tokens, 50)
    self.assertEqual(call.output_tokens, 10)
    self.assertEqual(call.thinking_tokens, 2)
    self.assertEqual(call.cached_tokens, 10)
    self.assertEqual(call.total_tokens, 62)

  def test_gpt4v_retry(self):
    gpt4v = infer.Gpt4Wrapper(model_name="gpt-4-turbo-2024-04-09")

    mock_429_response = requests.Response()
    mock_429_response.status_code = 429
    mock_429_response._content = (
        b'{"error": {"message": "Error 429: rate limit reached."}}'
    )

    mock_200_response = requests.Response()
    mock_200_response.status_code = 200
    mock_200_response._content = (
        b'{"choices": [{"message": {"content": "ok."}}]}'
    )
    self.mock_post.side_effect = [mock_429_response, mock_200_response]

    gpt4v.predict_mm("fake prompt", [])
    self.mock_sleep.assert_called_once()


if __name__ == "__main__":
  absltest.main()
