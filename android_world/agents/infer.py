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

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import time
from typing import Any, Optional

from android_world.agents import llm_usage
from google import genai
from google.genai import types as genai_types
import numpy as np
from PIL import Image
import requests


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)  # pyrefly: ignore[bad-assignment]
  return image_to_jpeg_bytes(image)  # pyrefly: ignore[bad-argument-type]


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = [
    genai_types.SafetySetting(
        category=category,
        threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
    )
    for category in (
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini interface, using the google-genai SDK and the Gemini API key."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
      seed: int | None = None,
      thinking_level: str | None = None,
  ):
    """Initializes the wrapper.

    Args:
      model_name: Gemini model id, e.g. 'gemini-3.5-flash-lite'.
      max_retry: Max attempts per request.
      temperature: Sampling temperature.
      top_p: Nucleus sampling parameter.
      enable_safety_checks: If False, all safety filters are set to BLOCK_NONE.
      seed: Sampling seed. Best-effort determinism only; None for no seed.
      thinking_level: Optional thinking level for models that support it
        (e.g. 'low', 'high'). None keeps the model default.
    """
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    if not model_name:
      raise ValueError('model_name is required.')
    self.client = genai.Client(api_key=os.environ['GCP_API_KEY'])
    self.model_name = model_name
    self.temperature = temperature
    self.top_p = top_p
    self.seed = seed
    self.thinking_level = thinking_level
    self.config = genai_types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        safety_settings=None
        if enable_safety_checks
        else SAFETY_SETTINGS_BLOCK_NONE,
        thinking_config=genai_types.ThinkingConfig(
            thinking_level=thinking_level.upper()  # pyrefly: ignore[bad-argument-type]
        )
        if thinking_level
        else None,
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def _config_for_call(
      self,
      enable_safety_checks: bool = True,
      generation_config: genai_types.GenerateContentConfigOrDict | None = None,
      safety_settings: list[genai_types.SafetySetting] | None = None,
  ) -> genai_types.GenerateContentConfig:
    """Merges per-call overrides into the default config."""
    update: dict[str, Any] = {}
    if generation_config is not None:
      if isinstance(generation_config, genai_types.GenerateContentConfig):
        generation_config = generation_config.model_dump(exclude_none=True)
      update.update(generation_config)  # pyrefly: ignore[no-matching-overload]
    if not enable_safety_checks:
      update['safety_settings'] = SAFETY_SETTINGS_BLOCK_NONE
    if safety_settings is not None:
      update['safety_settings'] = safety_settings
    if not update:
      return self.config
    return genai_types.GenerateContentConfig(
        **(self.config.model_dump(exclude_none=True) | update)
    )

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: genai_types.GenerateContentConfigOrDict | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != genai_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def _call_with_retries(
      self,
      contents: list[Any],
      config: genai_types.GenerateContentConfig,
      num_images: int,
      prompt_chars: int,
  ) -> tuple[Optional[str], Any]:
    """Calls the model, retrying on errors and empty responses.

    Every request is recorded in the usage tracker, including tokens billed
    for attempts that failed after the model responded.

    Args:
      contents: Model input.
      config: Generation config.
      num_images: Number of images in `contents`, for usage tracking.
      prompt_chars: Total characters of text in `contents`.

    Returns:
      The output text (None if every attempt failed) and the last raw response.
    """
    counter = self.max_retry
    retry_delay = 1.0
    output = None
    usage: dict[str, int] = {}
    attempts = 0
    start = time.time()
    text = None
    while counter > 0:
      attempts += 1
      try:
        output = self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config
        )
        llm_usage.add_usage(usage, llm_usage.gemini_usage(output))
        text = output.text
        if text is None:
          raise ValueError(f'Empty response from LLM: {output}')
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        text = None
        counter -= 1
        print(f'Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    llm_usage.get_tracker().record(
        self.model_name,
        latency_s=time.time() - start,
        attempts=attempts,
        success=text is not None,
        num_images=num_images,
        prompt_chars=prompt_chars,
        output_chars=len(text or ''),
        **usage,
    )
    return text, output

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: genai_types.GenerateContentConfigOrDict | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    config = self._config_for_call(enable_safety_checks, generation_config)
    contents = [text_prompt] + [Image.fromarray(image) for image in images]
    text, output = self._call_with_retries(
        contents, config, num_images=len(images), prompt_chars=len(text_prompt)
    )
    if text is not None:
      return text, True, output
    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: list[str | np.ndarray | Image.Image] | Any,
      safety_settings: list[genai_types.SafetySetting] | None = None,
      generation_config: genai_types.GenerateContentConfigOrDict | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    else:
      contents = [contents]
    config = self._config_for_call(
        generation_config=generation_config, safety_settings=safety_settings
    )
    text, response = self._call_with_retries(
        contents,
        config,
        num_images=sum(isinstance(c, Image.Image) for c in contents),
        prompt_chars=sum(len(c) for c in contents if isinstance(c, str)),
    )
    if text is None:
      raise RuntimeError(f'Error calling LLM. {response}.')
    return text, response

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> list[str | Image.Image]:
    """Converts numpy arrays in a content list to PIL images."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
    seed: Optional sampling seed (best-effort determinism).
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
      seed: int | None = None,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name
    self.model_name = model_name
    self.seed = seed

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 1000,
    }
    if self.seed is not None:
      payload['seed'] = self.seed

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {  # pyrefly: ignore[bad-assignment]
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    usage: dict[str, int] = {}
    attempts = 0
    start = time.time()

    def record(success: bool, output_chars: int = 0) -> None:
      llm_usage.get_tracker().record(
          self.model,
          latency_s=time.time() - start,
          attempts=attempts,
          success=success,
          num_images=len(images),
          prompt_chars=len(text_prompt),
          output_chars=output_chars,
          **usage,
      )

    while counter > 0:
      attempts += 1
      try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
        )
        response_json = response.json()
        llm_usage.add_usage(usage, llm_usage.openai_usage(response_json))
        if response.ok and 'choices' in response_json:
          text = response_json['choices'][0]['message']['content']
          record(True, len(text or ''))
          return (text, None, response)
        print(
            'Error calling OpenAI API with error message: '
            + response_json['error']['message']
        )
        counter -= 1
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    record(False)
    return ERROR_CALLING_LLM, None, None
