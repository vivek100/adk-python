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

"""Tests for ArrayIteratorAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator

from google.adk.agents.array_iterator_agent import ArrayIteratorAgent, _get_nested_value, _set_nested_value
from google.adk.agents.base_agent import BaseAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.state import State


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, name: str):
        super().__init__(name=name)
        self.call_count = 0
        self.yielded_events = []
    
    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        self.call_count += 1
        # Create a mock event
        event = MagicMock(spec=Event)
        event.content = f"processed_{ctx.session.state.get('current_item', 'unknown')}"
        event.actions = MagicMock(spec=EventActions)
        event.actions.escalate = False
        self.yielded_events.append(event)
        yield event
    
    async def _run_live_impl(self, ctx) -> AsyncGenerator[Event, None]:
        yield  # AsyncGenerator requires at least one yield


class MockEscalatingAgent(BaseAgent):
    """Mock agent that escalates after processing first item."""
    
    def __init__(self, name: str):
        super().__init__(name=name)
        self.call_count = 0
    
    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        self.call_count += 1
        event = MagicMock(spec=Event)
        event.content = f"processed_{ctx.session.state.get('current_item', 'unknown')}"
        event.actions = MagicMock(spec=EventActions)
        # Escalate if this is the second call
        event.actions.escalate = self.call_count >= 2
        yield event
    
    async def _run_live_impl(self, ctx) -> AsyncGenerator[Event, None]:
        yield  # AsyncGenerator requires at least one yield


class TestNestedKeyUtils:
    """Test the nested key utility functions."""
    
    def test_get_nested_value_simple(self):
        data = {"key": "value"}
        assert _get_nested_value(data, "key") == "value"
    
    def test_get_nested_value_nested(self):
        data = {"user": {"profile": {"name": "John"}}}
        assert _get_nested_value(data, "user.profile.name") == "John"
    
    def test_get_nested_value_missing_key(self):
        data = {"key": "value"}
        with pytest.raises(KeyError, match="Key path 'missing' not found"):
            _get_nested_value(data, "missing")
    
    def test_get_nested_value_missing_nested_key(self):
        data = {"user": {"profile": {}}}
        with pytest.raises(KeyError, match="Key path 'user.profile.name' not found"):
            _get_nested_value(data, "user.profile.name")
    
    def test_get_nested_value_non_dict(self):
        data = {"user": "not_a_dict"}
        with pytest.raises(TypeError, match="Cannot access key 'profile' on non-dict value"):
            _get_nested_value(data, "user.profile.name")
    
    def test_get_nested_value_empty_key(self):
        data = {"key": "value"}
        with pytest.raises(KeyError, match="Key path cannot be empty"):
            _get_nested_value(data, "")
    
    def test_set_nested_value_simple(self):
        data = {}
        _set_nested_value(data, "key", "value")
        assert data == {"key": "value"}
    
    def test_set_nested_value_nested(self):
        data = {}
        _set_nested_value(data, "user.profile.name", "John")
        assert data == {"user": {"profile": {"name": "John"}}}
    
    def test_set_nested_value_existing_path(self):
        data = {"user": {"profile": {"age": 30}}}
        _set_nested_value(data, "user.profile.name", "John")
        assert data == {"user": {"profile": {"age": 30, "name": "John"}}}
    
    def test_set_nested_value_non_dict_conflict(self):
        data = {"user": "not_a_dict"}
        with pytest.raises(TypeError, match="Cannot set nested key on non-dict value"):
            _set_nested_value(data, "user.profile.name", "John")
    
    def test_set_nested_value_empty_key(self):
        data = {}
        with pytest.raises(ValueError, match="Key path cannot be empty"):
            _set_nested_value(data, "", "value")


class TestArrayIteratorAgent:
    """Test the ArrayIteratorAgent class."""
    
    def test_init_valid_single_agent(self):
        """Test initialization with a single sub-agent."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="items",
            sub_agents=[sub_agent]
        )
        assert agent.name == "iterator"
        assert agent.array_key == "items"
        assert agent.item_key == "current_item"  # default
        assert agent.output_key is None  # default
        assert len(agent.sub_agents) == 1
    
    def test_init_no_sub_agents(self):
        """Test initialization fails with no sub-agents."""
        with pytest.raises(ValueError, match="ArrayIteratorAgent requires exactly one sub-agent"):
            ArrayIteratorAgent(
                name="iterator",
                array_key="items",
                sub_agents=[]
            )
    
    def test_init_multiple_sub_agents(self):
        """Test initialization fails with multiple sub-agents."""
        sub_agent1 = MockAgent("sub_agent1")
        sub_agent2 = MockAgent("sub_agent2")
        
        with pytest.raises(ValueError, match="ArrayIteratorAgent accepts only one sub-agent, but 2 were provided"):
            ArrayIteratorAgent(
                name="iterator",
                array_key="items",
                sub_agents=[sub_agent1, sub_agent2]
            )
    
    def test_custom_configuration(self):
        """Test custom item_key and output_key configuration."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="data.items",
            item_key="current_data",
            output_key="results.processed",
            sub_agents=[sub_agent]
        )
        assert agent.array_key == "data.items"
        assert agent.item_key == "current_data"
        assert agent.output_key == "results.processed"

    @pytest.mark.asyncio
    async def test_run_async_simple_array(self):
        """Test basic array iteration."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="items",
            item_key="current_item",
            output_key="results",
            sub_agents=[sub_agent]
        )
        
        # Mock context
        ctx = MagicMock()
        state = State(
            value={"items": ["item1", "item2", "item3"]},
            delta={}
        )
        ctx.session.state = state
        
        # Run the agent
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        
        # Verify results
        assert len(events) == 3  # One event per item
        assert sub_agent.call_count == 3
        assert state["results"] == ["processed_item1", "processed_item2", "processed_item3"]
    
    @pytest.mark.asyncio
    async def test_run_async_nested_array(self):
        """Test nested array access."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="data.items",
            item_key="current_item",
            sub_agents=[sub_agent]
        )
        
        # Mock context with nested data
        ctx = MagicMock()
        state = State(
            value={"data": {"items": ["nested1", "nested2"]}},
            delta={}
        )
        ctx.session.state = state
        
        # Run the agent
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        
        # Verify results
        assert len(events) == 2
        assert sub_agent.call_count == 2
    
    @pytest.mark.asyncio
    async def test_run_async_missing_array_key(self):
        """Test handling of missing array key."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="missing_key",
            sub_agents=[sub_agent]
        )
        
        # Mock context
        ctx = MagicMock()
        state = State(value={}, delta={})
        ctx.session.state = state
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Array key 'missing_key' not found or invalid"):
            async for event in agent._run_async_impl(ctx):
                pass
    
    @pytest.mark.asyncio
    async def test_run_async_non_array_value(self):
        """Test handling of non-array value."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="not_array",
            sub_agents=[sub_agent]
        )
        
        # Mock context
        ctx = MagicMock()
        state = State(value={"not_array": "string_value"}, delta={})
        ctx.session.state = state
        
        # Should raise TypeError
        with pytest.raises(TypeError, match="Value at 'not_array' is not a list"):
            async for event in agent._run_async_impl(ctx):
                pass
    
    @pytest.mark.asyncio
    async def test_run_async_empty_array(self):
        """Test handling of empty array."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="items",
            sub_agents=[sub_agent]
        )
        
        # Mock context
        ctx = MagicMock()
        state = State(value={"items": []}, delta={})
        ctx.session.state = state
        
        # Run the agent
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        
        # Should process no items
        assert len(events) == 0
        assert sub_agent.call_count == 0
    
    @pytest.mark.asyncio
    async def test_run_async_escalation_handling(self):
        """Test that escalation stops iteration."""
        sub_agent = MockEscalatingAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="items",
            sub_agents=[sub_agent]
        )
        
        # Mock context
        ctx = MagicMock()
        state = State(
            value={"items": ["item1", "item2", "item3", "item4"]},
            delta={}
        )
        ctx.session.state = state
        
        # Run the agent
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        
        # Should stop after second item due to escalation
        assert len(events) == 2
        assert sub_agent.call_count == 2
    
    @pytest.mark.asyncio
    async def test_run_async_state_restoration(self):
        """Test that item_key is properly restored."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="items",
            item_key="test_key",
            sub_agents=[sub_agent]
        )
        
        # Mock context with existing value for item_key
        ctx = MagicMock()
        state = State(
            value={"items": ["item1"], "test_key": "original_value"},
            delta={}
        )
        ctx.session.state = state
        
        # Run the agent
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        
        # Original value should be restored
        assert state["test_key"] == "original_value"
    
    @pytest.mark.asyncio
    async def test_run_async_no_output_key(self):
        """Test iteration without collecting results."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="items",
            # No output_key specified
            sub_agents=[sub_agent]
        )
        
        # Mock context
        ctx = MagicMock()
        state = State(
            value={"items": ["item1", "item2"]},
            delta={}
        )
        ctx.session.state = state
        
        # Run the agent
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        
        # No results should be stored
        assert len(events) == 2
        assert "results" not in state
    
    @pytest.mark.asyncio
    async def test_run_live_impl_not_supported(self):
        """Test that live mode raises NotImplementedError."""
        sub_agent = MockAgent("sub_agent")
        agent = ArrayIteratorAgent(
            name="iterator",
            array_key="items",
            sub_agents=[sub_agent]
        )
        
        ctx = MagicMock()
        
        with pytest.raises(NotImplementedError, match="Live mode is not supported"):
            async for event in agent._run_live_impl(ctx):
                pass 