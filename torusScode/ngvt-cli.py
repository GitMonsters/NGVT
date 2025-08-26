#!/usr/bin/env python3
"""
NGVT CLI - Command Line Interface for Nonlinear Geometric Vortexing Torus Code Generation

This script provides a command-line interface to the NGVT code generation system,
allowing users to generate code snippets from text prompts using the Vortex Code generator.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from vortex_code_simple import SimpleVortexGenerator
except ImportError as e:
    print(f"❌ Error importing SimpleVortexGenerator: {e}")
    print("Please ensure you're running this script from the torusScode directory.")
    sys.exit(1)


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="NGVT CLI - Generate code using the Vortex Code system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Write a function to calculate factorial"
  %(prog)s "def add(a, b):" --max-tokens 256
  %(prog)s "Implement binary search" --output result.py
  %(prog)s --list-templates
  %(prog)s --interactive

For more information about NGVT, visit: https://github.com/GitMonsters/NGVT
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Code generation prompt (e.g., "Write a function to calculate factorial")'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum number of tokens to generate (default: 512)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file to save the generated code (default: print to stdout)'
    )
    
    parser.add_argument(
        '--list-templates',
        action='store_true',
        help='List available code templates and exit'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode for multiple code generations'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='NGVT CLI 1.0.0'
    )
    
    return parser


def list_templates(generator):
    """List available templates in the generator."""
    print("🌀 Available NGVT Code Templates:")
    print("=" * 50)
    
    templates = generator.templates
    if not templates:
        print("No templates found.")
        return
    
    for i, (key, template) in enumerate(templates.items(), 1):
        print(f"{i:2d}. {key}")
        # Show first line of template as preview
        first_line = template.split('\n')[0]
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        print(f"    Preview: {first_line}")
        print()


def interactive_mode(generator):
    """Start interactive mode for continuous code generation."""
    print("🌀 NGVT Interactive Code Generation")
    print("=" * 50)
    print("Enter code prompts (type 'quit', 'exit', or Ctrl+C to exit)")
    print("Commands:")
    print("  - 'help': Show this help")
    print("  - 'templates': List available templates")
    print("  - 'clear': Clear screen")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\n🔮 Enter prompt: ").strip()
            
            if not prompt:
                continue
                
            if prompt.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            if prompt.lower() == 'help':
                print("\nCommands:")
                print("  - 'help': Show this help")
                print("  - 'templates': List available templates")
                print("  - 'clear': Clear screen")
                print("  - 'quit' or 'exit': Exit interactive mode")
                continue
                
            if prompt.lower() == 'templates':
                list_templates(generator)
                continue
                
            if prompt.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            print("\n⚡ Generating code...")
            code = generator.generate_code(prompt)
            print("\n📝 Generated Code:")
            print("-" * 40)
            print(code)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle list templates
    if args.list_templates:
        try:
            generator = SimpleVortexGenerator()
            list_templates(generator)
        except Exception as e:
            print(f"❌ Error initializing generator: {e}")
            sys.exit(1)
        return
    
    # Handle interactive mode
    if args.interactive:
        try:
            generator = SimpleVortexGenerator()
            interactive_mode(generator)
        except Exception as e:
            print(f"❌ Error initializing generator: {e}")
            sys.exit(1)
        return
    
    # Handle regular code generation
    if not args.prompt:
        parser.error("Please provide a prompt or use --interactive mode")
    
    try:
        # Initialize the generator
        print("🌀 Initializing NGVT Code Generator...")
        generator = SimpleVortexGenerator()
        
        # Generate code
        print(f"⚡ Generating code for: '{args.prompt}'")
        code = generator.generate_code(args.prompt, max_tokens=args.max_tokens)
        
        # Output the result
        if args.output:
            with open(args.output, 'w') as f:
                f.write(code)
            print(f"✅ Code saved to: {args.output}")
        else:
            print("\n📝 Generated Code:")
            print("=" * 50)
            print(code)
            print("=" * 50)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()