# Copyright 2025 Google LLC
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

"""Array iterator agent implementation."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Optional

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


def _get_nested_value(data: dict[str, Any], key_path: str) -> Any:
  """Get value from nested dictionary using dot notation.
  
  Args:
    data: The dictionary to search in.
    key_path: The key path using dot notation (e.g., 'user.profile.name').
    
  Returns:
    The value at the specified path.
    
  Raises:
    KeyError: If the key path is not found.
    TypeError: If trying to access a key on a non-dict value.
  """
  if not key_path:
    raise KeyError("Key path cannot be empty")
    
  keys = key_path.split('.')
  current = data
  
  for i, key in enumerate(keys):
    if not isinstance(current, dict):
      path_so_far = '.'.join(keys[:i])
      raise TypeError(
          f"Cannot access key '{key}' on non-dict value at path '{path_so_far}'"
      )
    
    if key not in current:
      path_so_far = '.'.join(keys[:i+1])
      raise KeyError(f"Key path '{path_so_far}' not found")
    
    current = current[key]
  
  return current


def _set_nested_value(data: dict[str, Any], key_path: str, value: Any) -> None:
  """Set value in nested dictionary using dot notation.
  
  Args:
    data: The dictionary to modify.
    key_path: The key path using dot notation (e.g., 'user.profile.name').
    value: The value to set.
    
  Raises:
    ValueError: If the key path is invalid.
    TypeError: If trying to set a key on a non-dict value.
  """
  if not key_path:
    raise ValueError("Key path cannot be empty")
    
  keys = key_path.split('.')
  current = data
  
  # Navigate to the parent of the final key
  for i, key in enumerate(keys[:-1]):
    if key not in current:
      current[key] = {}
    elif not isinstance(current[key], dict):
      path_so_far = '.'.join(keys[:i+1])
      raise TypeError(
          f"Cannot set nested key on non-dict value at path '{path_so_far}'"
      )
    current = current[key]
  
  # Set the final value
  current[keys[-1]] = value


class ArrayIteratorAgent(BaseAgent):
  """Agent that iterates over an array and applies a single sub-agent to each item.
  
  This agent focuses solely on iteration - it takes an array from session state,
  applies one sub-agent to each item, and optionally collects the results.
  
  Example:
    ```python
    processor = ArrayIteratorAgent(
        name="document_processor",
        array_key="documents",  # Can be nested: "user.documents"
        item_key="current_doc",
        output_key="processed_results",
        sub_agents=[document_analyzer]  # Exactly one sub-agent
    )
    ```
  """
  
  model_config = ConfigDict(extra="forbid", exclude_none=True)
  
  array_key: str = Field(..., description="Path to array in session state (supports dot notation)")
  item_key: str = Field(default="current_item", description="Key to store current item in session state")
  output_key: Optional[str] = Field(default=None, description="Key to store collected results array")
  
  @model_validator(mode='after')
  def _validate_single_sub_agent(self) -> 'ArrayIteratorAgent':
    """Validate that exactly one sub-agent is provided."""
    if len(self.sub_agents) == 0:
      raise ValueError("ArrayIteratorAgent requires exactly one sub-agent")
    
    if len(self.sub_agents) > 1:
      raise ValueError(
          f"ArrayIteratorAgent accepts only one sub-agent, but {len(self.sub_agents)} were provided. "
          f"If you need multiple agents, use SequentialAgent or ParallelAgent as the single sub-agent."
      )
    
    return self
  
  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Execute the array iteration logic."""
    
    # Get the array from session state
    try:
      state_dict = ctx.session.state.to_dict()
      array_data = _get_nested_value(state_dict, self.array_key)
    except (KeyError, TypeError) as e:
      logger.error(f"Failed to get array from key '{self.array_key}': {e}")
      raise ValueError(f"Array key '{self.array_key}' not found or invalid: {e}")
    
    # Validate that we have an array
    if not isinstance(array_data, list):
      raise TypeError(
          f"Value at '{self.array_key}' is not a list. Got {type(array_data).__name__}"
      )
    
    if not array_data:
      logger.info(f"Array at '{self.array_key}' is empty, skipping iteration")
      return
    
    logger.info(f"Starting iteration over {len(array_data)} items from '{self.array_key}'")
    
    # Collect results if output_key is specified
    results = [] if self.output_key else None
    sub_agent = self.sub_agents[0]
    
    # Store original item_key value to restore later
    original_item_value = ctx.session.state.get(self.item_key)
    
    try:
      # Iterate over each item in the array
      for i, item in enumerate(array_data):
        logger.debug(f"Processing item {i+1}/{len(array_data)}")
        
        # Inject current item into session state
        ctx.session.state[self.item_key] = item
        
        # Execute sub-agent for this item
        item_results = []
        async for event in sub_agent.run_async(ctx):
          yield event
          item_results.append(event)
        
        # Collect result if output_key is specified
        if self.output_key:
          # Get the last event's content as the result
          if item_results:
            last_event = item_results[-1]
            if hasattr(last_event, 'content') and last_event.content:
              results.append(last_event.content)
            else:
              results.append(None)
          else:
            results.append(None)
        
        # Check for escalation
        if item_results and any(event.actions.escalate for event in item_results):
          logger.info(f"Sub-agent escalated on item {i+1}, stopping iteration")
          break
    
    finally:
      # Restore original item_key value
      if original_item_value is not None:
        ctx.session.state[self.item_key] = original_item_value
      else:
        # Remove the item key if it wasn't there originally
        if self.item_key in ctx.session.state:
          del ctx.session.state[self.item_key]
    
    # Store results if output_key is specified
    if self.output_key and results is not None:
      try:
        # For simple keys, use direct assignment
        if '.' not in self.output_key:
          ctx.session.state[self.output_key] = results
        else:
          # For nested keys, we need to work with the state dict
          state_dict = ctx.session.state.to_dict()
          _set_nested_value(state_dict, self.output_key, results)
          # Update the session state with the modified dict
          ctx.session.state.update(state_dict)
        logger.info(f"Stored {len(results)} results in '{self.output_key}'")
      except (ValueError, TypeError) as e:
        logger.error(f"Failed to store results in '{self.output_key}': {e}")
        raise ValueError(f"Cannot store results in output key '{self.output_key}': {e}")
  
  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Live implementation for ArrayIteratorAgent."""
    raise NotImplementedError('Live mode is not supported for ArrayIteratorAgent yet.')
    yield  # AsyncGenerator requires having at least one yield statement 