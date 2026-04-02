"""Unit tests for langchain_ibm.utils module."""

import json

from langchain_ibm.utils import normalize_tool_arguments


def test_normalize_tool_arguments_valid_json_string() -> None:
    """Test normalizing valid JSON string arguments."""
    json_str = '{"location": "San Francisco", "unit": "celsius"}'
    result = normalize_tool_arguments(json_str)
    assert result == '{"location": "San Francisco", "unit": "celsius"}'
    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed["location"] == "San Francisco"


def test_normalize_tool_arguments_python_dict_string() -> None:
    """Test normalizing Python dict string with single quotes."""
    python_dict_str = "{'location': 'San Francisco', 'unit': 'celsius'}"
    result = normalize_tool_arguments(python_dict_str)
    assert result == '{"location": "San Francisco", "unit": "celsius"}'


def test_normalize_tool_arguments_extra_wrapping_quotes() -> None:
    """Test normalizing arguments with extra surrounding quotes."""
    wrapped_json = '"{\\"location\\": \\"San Francisco\\", \\"unit\\": \\"celsius\\"}"'
    result = normalize_tool_arguments(wrapped_json)
    assert result == '{"location": "San Francisco", "unit": "celsius"}'


def test_normalize_tool_arguments_nested_structures() -> None:
    """Test normalizing arguments with nested dict/list structures."""
    nested_str = '{"user": {"name": "John", "prefs": ["temp", "humidity"]}}'
    result = normalize_tool_arguments(nested_str)
    assert result == '{"user": {"name": "John", "prefs": ["temp", "humidity"]}}'


def test_normalize_tool_arguments_empty_dict() -> None:
    """Test normalizing empty dict arguments."""
    empty_dict = "{}"
    result = normalize_tool_arguments(empty_dict)
    assert result == "{}"


def test_normalize_tool_arguments_already_valid_json() -> None:
    """Test that already valid JSON is returned as-is."""
    valid_json = '{"key": "value", "number": 42, "bool": true}'
    result = normalize_tool_arguments(valid_json)
    assert result == '{"key": "value", "number": 42, "bool": true}'


def test_normalize_tool_arguments_special_characters() -> None:
    """Test normalizing arguments with special characters."""
    special_chars = '{"message": "Hello \\"world\\"!", "path": "C:\\\\\\\\Users"}'
    result = normalize_tool_arguments(special_chars)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert "message" in parsed


def test_normalize_tool_arguments_numbers_and_booleans() -> None:
    """Test normalizing arguments with various data types."""
    mixed_types = '{"temp": 25.5, "enabled": true, "count": 10, "data": null}'
    result = normalize_tool_arguments(mixed_types)
    parsed = json.loads(result)
    assert parsed["temp"] == 25.5
    assert parsed["enabled"] is True
    assert parsed["count"] == 10
    assert parsed["data"] is None


def test_normalize_tool_arguments_malformed_vllm_string() -> None:
    """Test normalizing malformed vLLM nested JSON strings with empty keys."""
    malformed_vllm = '"{\\n  \\"\\": {}\\n}"'
    result = normalize_tool_arguments(malformed_vllm)
    # Should return empty dict as fallback for invalid empty key pattern
    parsed = json.loads(result)
    assert parsed == {}


def test_normalize_tool_arguments_double_wrapped_json() -> None:
    """Test normalizing double-wrapped JSON string."""
    double_wrapped = '"{\\"name\\": \\"test\\"}"'
    result = normalize_tool_arguments(double_wrapped)
    parsed = json.loads(result)
    assert parsed == {"name": "test"}


def test_normalize_tool_arguments_triple_wrapped_json() -> None:
    """Test normalizing triple-wrapped JSON string."""
    triple_wrapped = '"\\"{\\\\\\"key\\\\\\": \\\\\\"value\\\\\\"}\\"" '
    result = normalize_tool_arguments(triple_wrapped)
    parsed = json.loads(result)
    assert "key" in parsed


def test_normalize_tool_arguments_trailing_garbage_double_quote_brace() -> None:
    """Test handling of trailing garbage: '{}'"}."""
    malformed = '"{}""}'
    result = normalize_tool_arguments(malformed)
    assert result == "{}"


def test_normalize_tool_arguments_trailing_garbage_quote_brace() -> None:
    """Test handling of trailing quote and brace."""
    malformed = '{}"}'
    result = normalize_tool_arguments(malformed)
    assert result == "{}"


def test_normalize_tool_arguments_trailing_garbage_multiple_braces() -> None:
    """Test handling of multiple trailing braces."""
    malformed = '{"key": "value"}}}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_trailing_garbage_text() -> None:
    """Test handling of trailing text garbage."""
    malformed = '{"key": "value"}abc'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_unbalanced_braces_missing_closing() -> None:
    """Test handling of missing closing brace."""
    malformed = '{"key": "value"'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_unbalanced_braces_extra_opening() -> None:
    """Test handling of extra opening brace."""
    malformed = '{{"key": "value"}'
    result = normalize_tool_arguments(malformed)
    # Successfully repairs by removing the extra brace
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_empty_string_input() -> None:
    """Test handling of empty string input."""
    result = normalize_tool_arguments("")
    assert result == "{}"


def test_normalize_tool_arguments_whitespace_only_input() -> None:
    """Test handling of whitespace-only input."""
    result = normalize_tool_arguments("   ")
    assert result == "{}"


def test_normalize_tool_arguments_completely_invalid_input() -> None:
    """Test handling of completely invalid input."""
    result = normalize_tool_arguments("not json at all")
    assert result == "{}"


def test_normalize_tool_arguments_null_value() -> None:
    """Test handling of null value."""
    # 'null' is not a dict/object, so it returns {} as fallback
    # since the function is designed for tool arguments (objects)
    result = normalize_tool_arguments("null")
    assert result == "{}"


def test_normalize_tool_arguments_array_input() -> None:
    """Test handling of JSON array."""
    result = normalize_tool_arguments("[1, 2, 3]")
    parsed = json.loads(result)
    assert parsed == [1, 2, 3]


def test_normalize_tool_arguments_trailing_comma() -> None:
    """Test handling of trailing comma (common LLM error)."""
    malformed = '{"key": "value",}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_leading_garbage() -> None:
    """Test handling of leading garbage before JSON."""
    malformed = 'garbage{"key": "value"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_unicode_characters() -> None:
    """Test handling of unicode characters."""
    unicode_str = '{"message": "Hello 世界", "emoji": "🎉"}'
    result = normalize_tool_arguments(unicode_str)
    parsed = json.loads(result)
    assert parsed["message"] == "Hello 世界"
    assert parsed["emoji"] == "🎉"


def test_normalize_tool_arguments_very_nested_structure() -> None:
    """Test deeply nested JSON structure."""
    nested = '{"a": {"b": {"c": {"d": {"e": "value"}}}}}'
    result = normalize_tool_arguments(nested)
    parsed = json.loads(result)
    assert parsed["a"]["b"]["c"]["d"]["e"] == "value"


def test_normalize_tool_arguments_large_json_object() -> None:
    """Test large JSON object."""
    large_obj = {f"key_{i}": f"value_{i}" for i in range(100)}
    json_str = json.dumps(large_obj)
    result = normalize_tool_arguments(json_str)
    parsed = json.loads(result)
    assert len(parsed) == 100


def test_normalize_tool_arguments_mixed_quotes_in_values() -> None:
    """Test JSON with mixed quotes in values."""
    mixed = '{"message": "He said \\"hello\\" to me"}'
    result = normalize_tool_arguments(mixed)
    parsed = json.loads(result)
    assert "hello" in parsed["message"]


def test_normalize_tool_arguments_newlines_in_json() -> None:
    """Test JSON with newlines."""
    with_newlines = '{\n  "key": "value",\n  "number": 42\n}'
    result = normalize_tool_arguments(with_newlines)
    parsed = json.loads(result)
    assert parsed == {"key": "value", "number": 42}


def test_normalize_tool_arguments_tabs_in_json() -> None:
    """Test JSON with tabs."""
    with_tabs = '{\t"key":\t"value"\t}'
    result = normalize_tool_arguments(with_tabs)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_boolean_values() -> None:
    """Test JSON with boolean values."""
    booleans = '{"true_val": true, "false_val": false}'
    result = normalize_tool_arguments(booleans)
    parsed = json.loads(result)
    assert parsed["true_val"] is True
    assert parsed["false_val"] is False


def test_normalize_tool_arguments_numeric_edge_cases() -> None:
    """Test JSON with numeric edge cases."""
    numbers = '{"zero": 0, "negative": -42, "float": 3.14, "exp": 1e10}'
    result = normalize_tool_arguments(numbers)
    parsed = json.loads(result)
    assert parsed["zero"] == 0
    assert parsed["negative"] == -42
    assert parsed["float"] == 3.14


# Edge Cases - Critical real-world LLM hallucination scenarios


def test_normalize_tool_arguments_mixed_quote_types() -> None:
    """Test mixed single and double quotes (common LLM error)."""
    # LLMs sometimes mix quote types
    malformed = """{"name": 'John', 'age': "30"}"""
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert "name" in parsed or "age" in parsed or parsed == {}


def test_normalize_tool_arguments_concatenated_json_objects() -> None:
    """Test multiple JSON objects concatenated."""
    # LLM continues generating and concatenates multiple objects
    malformed = '{"key1": "value1"}{"key2": "value2"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    # Concatenated objects are converted into an array
    assert parsed == [{"key1": "value1"}, {"key2": "value2"}]


def test_normalize_tool_arguments_json_with_comments() -> None:
    """Test JSON with comments (LLMs add explanations)."""
    # LLM adds comments which are invalid in JSON
    malformed = '{"key": "value"} // This is the result'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_json_with_markdown_code_block() -> None:
    """Test JSON wrapped in markdown code block."""
    # LLM wraps response in markdown
    malformed = '```json\n{"key": "value"}\n```'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_json_with_text_prefix() -> None:
    """Test JSON with explanatory text prefix."""
    # LLM adds explanation before JSON
    malformed = 'Here is the result: {"key": "value"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_json_with_text_suffix() -> None:
    """Test JSON with explanatory text suffix."""
    # LLM adds explanation after JSON
    malformed = '{"key": "value"} - This is the answer'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_single_quoted_json() -> None:
    """Test JSON with single quotes instead of double quotes."""
    # LLM uses Python-style single quotes
    malformed = "{'key': 'value', 'number': 42}"
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value", "number": 42}


def test_normalize_tool_arguments_unquoted_keys() -> None:
    """Test JSON with unquoted keys (JavaScript-style)."""
    # LLM generates JavaScript-style object literal
    malformed = "{key: 'value', number: 42}"
    result = normalize_tool_arguments(malformed)
    # This is very hard to parse reliably, should fallback to {}
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_escaped_quotes_error() -> None:
    """Test incorrectly escaped quotes."""
    # LLM escapes quotes incorrectly
    malformed = '{"message": "He said \\"hello\\\\"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert "message" in parsed


def test_normalize_tool_arguments_missing_comma_between_fields() -> None:
    """Test missing comma between fields."""
    # LLM forgets comma separator
    malformed = '{"key1": "value1" "key2": "value2"}'
    result = normalize_tool_arguments(malformed)
    # This is invalid and hard to repair, should fallback
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_extra_commas() -> None:
    """Test extra commas between fields."""
    # LLM adds extra commas
    malformed = '{"key1": "value1",, "key2": "value2"}'
    result = normalize_tool_arguments(malformed)
    # Should handle or fallback gracefully
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_trailing_text_after_complete_json() -> None:
    """Test complete JSON followed by random text."""
    # LLM continues generating after valid JSON
    malformed = '{"key": "value"} and some more text here'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_json_array_instead_of_object() -> None:
    """Test when LLM returns array instead of object."""
    # LLM returns array when object expected
    malformed = '["value1", "value2"]'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == ["value1", "value2"]


def test_normalize_tool_arguments_nested_quotes_confusion() -> None:
    """Test nested quotes causing confusion."""
    # LLM gets confused with nested quotes
    # This is actually invalid JSON that can't be reliably parsed
    malformed = '{"outer": "{"inner": "value"}"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    # Should fallback to {} for unparseable input
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_boolean_as_string() -> None:
    """Test boolean values as strings."""
    # LLM returns "true"/"false" as strings instead of booleans
    malformed = '{"enabled": "true", "disabled": "false"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    # Should parse as strings (which is what LLM intended)
    assert parsed["enabled"] == "true"
    assert parsed["disabled"] == "false"


# Additional Edge Cases - Less common but important scenarios


def test_normalize_tool_arguments_bom_character() -> None:
    """Test JSON with BOM (Byte Order Mark) character."""
    # Some systems add BOM to UTF-8 files
    malformed = '\ufeff{"key": "value"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert parsed == {"key": "value"}


def test_normalize_tool_arguments_control_characters() -> None:
    """Test JSON with control characters."""
    # LLM output may contain control characters
    malformed = '{"key": "value\x00\x01\x02"}'
    result = normalize_tool_arguments(malformed)
    # Should handle gracefully
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_extremely_nested_wrapping() -> None:
    """Test extremely nested JSON string wrapping (5+ levels)."""
    # Some models wrap JSON strings multiple times
    # This is extremely rare and may not fully unwrap all levels
    malformed = (
        '"\\"\\"\\\\\\"{\\\\\\\\\\\\\\"key\\\\\\\\\\\\\\":'
        '\\\\\\\\\\\\\\"value\\\\\\\\\\\\\\"}\\\\\\"\\"\\""'
    )
    result = normalize_tool_arguments(malformed)
    # Should handle gracefully - may return string or fallback to {}
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list, str))


def test_normalize_tool_arguments_json_with_null_bytes() -> None:
    """Test JSON with null bytes."""
    # Rare but possible in some edge cases
    malformed = '{"key": "val\x00ue"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_json_with_surrogate_pairs() -> None:
    """Test JSON with Unicode surrogate pairs."""
    # Emoji and special Unicode characters
    malformed = '{"emoji": "\\uD83D\\uDE00"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert "emoji" in parsed


def test_normalize_tool_arguments_json_with_scientific_notation() -> None:
    """Test JSON with scientific notation edge cases."""
    # Various scientific notation formats
    malformed = '{"small": 1e-100, "large": 1e100, "negative": -1.5e-10}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert "small" in parsed
    assert "large" in parsed


def test_normalize_tool_arguments_json_with_infinity() -> None:
    """Test JSON with Infinity (invalid in JSON spec)."""
    # LLM might generate Infinity which is invalid JSON
    malformed = '{"value": Infinity}'
    result = normalize_tool_arguments(malformed)
    # Should fallback gracefully
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_json_with_nan() -> None:
    """Test JSON with NaN (invalid in JSON spec)."""
    # LLM might generate NaN which is invalid JSON
    malformed = '{"value": NaN}'
    result = normalize_tool_arguments(malformed)
    # Should fallback gracefully
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_json_with_undefined() -> None:
    """Test JSON with undefined (JavaScript concept)."""
    # LLM trained on JavaScript might use undefined
    malformed = '{"value": undefined}'
    result = normalize_tool_arguments(malformed)
    # Should fallback gracefully
    parsed = json.loads(result)
    assert isinstance(parsed, (dict, list))


def test_normalize_tool_arguments_json_with_plus_sign() -> None:
    """Test JSON with explicit plus sign on numbers."""
    # LLM might add plus sign (invalid in JSON but valid in Python)
    malformed = '{"value": +42}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    # Python literal_eval can handle this
    assert parsed["value"] == 42


def test_normalize_tool_arguments_extremely_long_string() -> None:
    """Test JSON with extremely long string value."""
    # Test with very long string (10000 chars)
    long_value = "x" * 10000
    malformed = f'{{"key": "{long_value}"}}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert len(parsed["key"]) == 10000


def test_normalize_tool_arguments_deeply_nested_object() -> None:
    """Test extremely deeply nested object (50+ levels)."""
    # Create deeply nested structure
    nested = '{"a":' * 50 + "{}" + "}" * 50
    result = normalize_tool_arguments(nested)
    parsed = json.loads(result)
    # Should parse successfully
    assert isinstance(parsed, dict)


def test_normalize_tool_arguments_json_with_duplicate_keys() -> None:
    """Test JSON with duplicate keys."""
    # LLM might generate duplicate keys
    malformed = '{"key": "value1", "key": "value2"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    # JSON spec: last value wins
    assert parsed["key"] == "value2"


def test_normalize_tool_arguments_json_with_empty_key() -> None:
    """Test JSON with empty string as key."""
    # LLM might use empty string as key (valid in JSON)
    malformed = '{"": "value", "normal": "data"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    # Both keys should be preserved (empty string key is valid)
    assert parsed[""] == "value"
    assert parsed["normal"] == "data"


def test_normalize_tool_arguments_json_with_numeric_keys() -> None:
    """Test JSON with numeric keys (as strings)."""
    # LLM might use numbers as keys
    malformed = '{"123": "value", "456": "data"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    assert "123" in parsed


def test_normalize_tool_arguments_json_with_special_key_names() -> None:
    """Test JSON with special characters in key names."""
    # Keys with special characters (valid in JSON)
    malformed = '{"key-with-dash": "v1", "key.with.dot": "v2", "key:with:colon": "v3"}'
    result = normalize_tool_arguments(malformed)
    parsed = json.loads(result)
    # All keys should be preserved
    assert parsed["key-with-dash"] == "v1"
    assert parsed["key.with.dot"] == "v2"
    assert parsed["key:with:colon"] == "v3"
