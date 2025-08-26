#!/usr/bin/env python3
"""
Simple test script to verify NGVT CLI functionality
"""

import subprocess
import sys
import os
import tempfile

def run_cli_command(args):
    """Run the CLI with given arguments and return output."""
    cmd = [sys.executable, "ngvt-cli.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    return result.returncode, result.stdout, result.stderr

def test_cli():
    """Test basic CLI functionality."""
    print("🧪 Testing NGVT CLI...")
    
    # Test 1: Help command
    print("1. Testing --help...")
    returncode, stdout, stderr = run_cli_command(["--help"])
    assert returncode == 0, f"Help command failed: {stderr}"
    assert "NGVT CLI" in stdout, "Help output doesn't contain expected text"
    print("   ✅ Help command works")
    
    # Test 2: List templates
    print("2. Testing --list-templates...")
    returncode, stdout, stderr = run_cli_command(["--list-templates"])
    assert returncode == 0, f"List templates failed: {stderr}"
    assert "factorial" in stdout, "Templates not listed properly"
    print("   ✅ Template listing works")
    
    # Test 3: Basic code generation
    print("3. Testing basic code generation...")
    returncode, stdout, stderr = run_cli_command(["Write a function to add two numbers"])
    assert returncode == 0, f"Code generation failed: {stderr}"
    assert "def" in stdout, "Generated code doesn't contain function definition"
    print("   ✅ Code generation works")
    
    # Test 4: Output to file
    print("4. Testing file output...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_file = f.name
    
    try:
        returncode, stdout, stderr = run_cli_command(["def factorial(n):", "--output", temp_file])
        assert returncode == 0, f"File output failed: {stderr}"
        
        with open(temp_file, 'r') as f:
            content = f.read()
        assert "def factorial" in content, "Generated file doesn't contain expected code"
        print("   ✅ File output works")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("\n🎉 All tests passed! NGVT CLI is working correctly.")

if __name__ == "__main__":
    test_cli()