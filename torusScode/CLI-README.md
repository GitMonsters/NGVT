# NGVT CLI - Command Line Interface

A command-line interface for the NGVT (Nonlinear Geometric Vortexing Torus) code generation system.

## Features

- **Simple Code Generation**: Generate code from natural language prompts
- **Template System**: Access 10+ built-in code templates (factorial, fibonacci, binary search, etc.)
- **Interactive Mode**: Continuous code generation session
- **File Output**: Save generated code directly to files
- **Help System**: Built-in help and template listing

## Installation

No additional installation required if you have the NGVT repository. The CLI uses the existing `SimpleVortexGenerator`.

## Usage

### Basic Code Generation

```bash
# Generate a function
python3 ngvt-cli.py "Write a function to calculate factorial"

# Complete a function definition  
python3 ngvt-cli.py "def add(a, b):"

# Generate and save to file
python3 ngvt-cli.py "Create a Stack class" --output stack.py
```

### List Available Templates

```bash
python3 ngvt-cli.py --list-templates
```

### Interactive Mode

```bash
python3 ngvt-cli.py --interactive
```

In interactive mode, you can:
- Enter multiple prompts
- Use commands like `help`, `templates`, `clear`
- Type `quit` or `exit` to leave

### Command Line Options

```
positional arguments:
  prompt                Code generation prompt

options:
  -h, --help            Show help message
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: 512)
  --output OUTPUT, -o OUTPUT
                        Output file to save generated code
  --list-templates      List available code templates
  --interactive, -i     Start interactive mode
  --version             Show version information
```

## Examples

### Generate Various Code Types

```bash
# Algorithms
python3 ngvt-cli.py "Implement binary search"
python3 ngvt-cli.py "Write bubble sort algorithm"

# Data structures
python3 ngvt-cli.py "Create a Queue class"
python3 ngvt-cli.py "Implement a stack"

# Utilities
python3 ngvt-cli.py "Function to reverse a string"
python3 ngvt-cli.py "Check if string is palindrome"
```

### Function Completion

```bash
# The CLI can complete partial function definitions
python3 ngvt-cli.py "def fibonacci(n):"
python3 ngvt-cli.py "def multiply(a, b):"
```

### Interactive Session Example

```bash
$ python3 ngvt-cli.py --interactive
🌀 NGVT Interactive Code Generation
==================================================
Enter code prompts (type 'quit', 'exit', or Ctrl+C to exit)

🔮 Enter prompt: def factorial(n):

⚡ Generating code...

📝 Generated Code:
----------------------------------------
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
----------------------------------------

🔮 Enter prompt: templates
🌀 Available NGVT Code Templates:
==================================================
 1. factorial
 2. fibonacci  
 3. binary search
 4. stack
 5. reverse string
 6. palindrome
 7. bubble sort
 8. add
 9. multiply
10. queue

🔮 Enter prompt: quit
👋 Goodbye!
```

## Available Templates

The CLI provides access to these built-in templates:

1. **factorial** - Recursive factorial function
2. **fibonacci** - Fibonacci sequence generator
3. **binary search** - Binary search algorithm
4. **stack** - Stack data structure class
5. **reverse string** - String reversal function
6. **palindrome** - Palindrome checker
7. **bubble sort** - Bubble sort algorithm
8. **add** - Addition function
9. **multiply** - Multiplication function
10. **queue** - Queue data structure class

## Technical Details

- Uses the `SimpleVortexGenerator` from the NGVT system
- Template-based generation for common patterns
- Fallback to generic code generation for unknown patterns
- No external dependencies beyond the base NGVT system

## Troubleshooting

**Module Import Error**: Make sure you're running the script from the `torusScode` directory.

**Permission Denied**: Make sure the script is executable:
```bash
chmod +x ngvt-cli.py
```

## Integration with Other Tools

The CLI can be easily integrated into development workflows:

```bash
# Generate code and edit
python3 ngvt-cli.py "Binary search function" --output search.py
nano search.py

# Generate multiple functions
python3 ngvt-cli.py "Stack class" --output data_structures.py
python3 ngvt-cli.py "Queue class" >> data_structures.py
```