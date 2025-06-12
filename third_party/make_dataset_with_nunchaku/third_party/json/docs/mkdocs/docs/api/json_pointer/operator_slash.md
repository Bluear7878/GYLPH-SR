# <small>nlohmann::json_pointer::</small>operator/

```cpp
// (1)
json_pointer operator/(const json_pointer& lhs, const json_pointer& rhs);

// (2)
json_pointer operator/(const json_pointer& lhs, string_t token);

// (3)
json_pointer operator/(const json_pointer& lhs, std::size_t array_idx);
```

1. create a new JSON pointer by appending the right JSON pointer at the end of the left JSON pointer
1. create a new JSON pointer by appending the unescaped token at the end of the JSON pointer
1. create a new JSON pointer by appending the array-index-token at the end of the JSON pointer

## Parameters

`lhs` (in)
: JSON pointer

`rhs` (in)
: JSON pointer to append

`token` (in)
: reference token to append

`array_idx` (in)
: array index to append

## Return value

1. a new JSON pointer with `rhs` appended to `lhs`
1. a new JSON pointer with unescaped `token` appended to `lhs`
1. a new JSON pointer with `array_idx` appended to `lhs`

## Complexity

1. Linear in the length of `lhs` and `rhs`.
1. Linear in the length of `lhs`.
1. Linear in the length of `lhs`.

## Examples

??? example

````
The example shows the usage of `operator/`.

```cpp
--8<-- "examples/json_pointer__operator_add_binary.cpp"
```

Output:

```json
--8<-- "examples/json_pointer__operator_add_binary.output"
```
````

## Version history

1. Added in version 3.6.0.
1. Added in version 3.6.0. Changed type of `token` to `string_t` in version 3.11.0.
1. Added in version 3.6.0.
