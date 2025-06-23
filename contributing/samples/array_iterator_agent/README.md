# ArrayIteratorAgent Sample

This sample demonstrates how to use the `ArrayIteratorAgent` for processing arrays of data with a single sub-agent.

## Overview

The `ArrayIteratorAgent` is designed to:
- **Iterate over arrays** in session state (supports nested keys with dot notation)
- **Apply a single sub-agent** to each array item
- **Collect results** optionally into an output array
- **Handle escalation** to stop processing when needed

## Key Features

### 🔧 **Single Sub-Agent Focus**
- Accepts exactly **one sub-agent** (enforced by validation)
- For complex processing, use `SequentialAgent` or `ParallelAgent` as the single sub-agent

### 🗂️ **Nested Key Support**
- Array key: `"documents"` or `"user.profile.documents"`
- Output key: `"results"` or `"processed.batch_results"`

### 📊 **Result Collection**
- Automatic collection of sub-agent results when `output_key` is specified
- Results stored as array in session state

### ⚡ **Escalation Handling**
- Stops iteration when sub-agent escalates
- Graceful cleanup of temporary state

## Usage Examples

### 1. Simple Document Processing

```python
from google.adk.agents import ArrayIteratorAgent, LlmAgent

# Document analyzer
analyzer = LlmAgent(
    name="document_analyzer",
    model=LiteLLMConnection(model_name="gpt-4o-mini"),
    instruction="Analyze document in {current_doc}",
    output_key="analysis"
)

# Array processor
processor = ArrayIteratorAgent(
    name="doc_processor",
    array_key="documents",           # Array in session state
    item_key="current_doc",          # Key for current item
    output_key="analyses",           # Collect results
    sub_agents=[analyzer]            # Single sub-agent
)
```

**Session State:**
```json
{
  "documents": [
    {"title": "Doc1", "content": "..."},
    {"title": "Doc2", "content": "..."}
  ]
}
```

**Result:**
```json
{
  "documents": [/* original docs */],
  "analyses": ["Analysis of Doc1", "Analysis of Doc2"]
}
```

### 2. Nested Data Processing

```python
# Process nested customer array
customer_processor = ArrayIteratorAgent(
    name="customer_processor",
    array_key="company.customers",        # Nested array access
    item_key="current_customer",
    output_key="company.processed",       # Nested output
    sub_agents=[customer_analyzer]
)
```

**Session State:**
```json
{
  "company": {
    "name": "TechCorp",
    "customers": [
      {"name": "Alice", "spend": 12000},
      {"name": "Bob", "spend": 7500}
    ]
  }
}
```

### 3. Complex Pipeline Processing

```python
# Multi-step pipeline as single sub-agent
pipeline = SequentialAgent(
    name="processing_pipeline",
    sub_agents=[extractor, validator, transformer]
)

# Use pipeline in array iterator
batch_processor = ArrayIteratorAgent(
    name="batch_processor",
    array_key="raw_data",
    item_key="current_item",
    output_key="processed_batch",
    sub_agents=[pipeline]  # Pipeline as single sub-agent
)
```

### 4. Without Result Collection

```python
# Process without collecting results
notifier = ArrayIteratorAgent(
    name="notification_sender",
    array_key="users",
    item_key="current_user",
    # No output_key - don't collect results
    sub_agents=[notification_agent]
)
```

## Configuration Options

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | ✅ | Agent name |
| `array_key` | `str` | ✅ | Path to array (supports `dot.notation`) |
| `item_key` | `str` | Optional | Key for current item (default: `"current_item"`) |
| `output_key` | `str` | Optional | Key to store results array |
| `sub_agents` | `list[BaseAgent]` | ✅ | **Exactly one sub-agent** |

## Error Handling

### Validation Errors
```python
# ❌ No sub-agents
ArrayIteratorAgent(name="bad", array_key="items", sub_agents=[])
# ValueError: ArrayIteratorAgent requires exactly one sub-agent

# ❌ Multiple sub-agents  
ArrayIteratorAgent(name="bad", array_key="items", sub_agents=[agent1, agent2])
# ValueError: ArrayIteratorAgent accepts only one sub-agent, but 2 were provided
```

### Runtime Errors
```python
# ❌ Missing array key
# ValueError: Array key 'missing_key' not found or invalid

# ❌ Non-array value
# TypeError: Value at 'not_array' is not a list. Got str
```

## Best Practices

### ✅ **Do:**
- Use single sub-agent pattern for focused iteration
- Leverage nested keys for complex data structures
- Use `SequentialAgent`/`ParallelAgent` as sub-agent for complex workflows
- Handle escalation gracefully in sub-agents

### ❌ **Don't:**
- Try to add multiple sub-agents directly
- Assume arrays are always non-empty
- Forget to handle missing keys in session state
- Mix iteration logic with processing logic

## Sample Data

The sample includes realistic test data:

```python
SAMPLE_DATA = {
    "documents": [/* document objects */],
    "company": {
        "customers": [/* customer objects */]
    },
    "raw_data": [/* processing items */],
    "items_to_process": [/* items with error cases */]
}
```

## Running the Sample

```bash
cd adk-python/contributing/samples/array_iterator_agent
python agent.py
```

This will show the different ArrayIteratorAgent configurations available.

## Related Agents

- **`LoopAgent`**: Fixed iteration count
- **`SequentialAgent`**: Sequential sub-agent execution  
- **`ParallelAgent`**: Parallel sub-agent execution

The `ArrayIteratorAgent` complements these by providing **data-driven iteration** over arrays. 