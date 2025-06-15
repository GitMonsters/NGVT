#!/usr/bin/env python3
"""
Simplified Vortex Code - AI Code Generation (Web Compatible)
"""

import json
import random
from pathlib import Path

class SimpleVortexGenerator:
    """Simplified Vortex Code generator for web deployment"""
    
    def __init__(self, model_path="."):
        self.model_path = Path(model_path)
        
        # Load templates
        template_file = self.model_path / "vortex_code_templates.json"
        if template_file.exists():
            with open(template_file, 'r') as f:
                self.templates = json.load(f)
        else:
            # Default templates if file doesn't exist
            self.templates = self._get_default_templates()
        
        print("🌀 Vortex Code initialized (Simplified Mode)")
        print(f"   Templates: {len(self.templates)} patterns")
    
    def _get_default_templates(self):
        """Default code templates"""
        return {
            "factorial": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
            
            "fibonacci": """def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
            
            "binary search": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
            
            "stack": """class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)""",
            
            "reverse string": """def reverse_string(s):
    return s[::-1]""",
            
            "palindrome": """def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]""",
            
            "bubble sort": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr""",
            
            "add": """def add(a, b):
    return a + b""",
            
            "multiply": """def multiply(a, b):
    return a * b""",
            
            "queue": """class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)"""
        }
    
    def generate_code(self, prompt, max_tokens=512):
        """Generate code based on prompt"""
        prompt_lower = prompt.lower()
        
        # Check for exact matches first
        for key, code in self.templates.items():
            if key in prompt_lower:
                return self._add_vortex_comment(code, prompt)
        
        # Check for function definitions
        if prompt.strip().startswith("def "):
            return self._complete_function(prompt)
        
        # Generate generic solution
        return self._generate_generic(prompt)
    
    def _complete_function(self, prompt):
        """Complete a function definition"""
        # Extract function name and parameters
        if "(" in prompt and ")" in prompt:
            func_parts = prompt.split("(")
            func_name = func_parts[0].replace("def", "").strip()
            params = func_parts[1].split(")")[0]
            
            # Generate appropriate function body based on name
            if "add" in func_name or "sum" in func_name:
                return f"{prompt}\n    return {' + '.join(p.strip() for p in params.split(',') if p.strip())}"
            elif "multiply" in func_name or "mult" in func_name:
                return f"{prompt}\n    return {' * '.join(p.strip() for p in params.split(',') if p.strip())}"
            elif "subtract" in func_name or "sub" in func_name:
                params_list = [p.strip() for p in params.split(',') if p.strip()]
                if len(params_list) >= 2:
                    return f"{prompt}\n    return {params_list[0]} - {params_list[1]}"
            elif "divide" in func_name or "div" in func_name:
                params_list = [p.strip() for p in params.split(',') if p.strip()]
                if len(params_list) >= 2:
                    return f"{prompt}\n    if {params_list[1]} != 0:\n        return {params_list[0]} / {params_list[1]}\n    return None"
            
        # Default implementation
        return f"{prompt}\n    # TODO: Implement function\n    pass"
    
    def _generate_generic(self, prompt):
        """Generate generic code solution"""
        prompt_lower = prompt.lower()
        
        # Common patterns
        if "function" in prompt_lower:
            if "add" in prompt_lower:
                return self.templates.get("add", "")
            elif "multiply" in prompt_lower:
                return self.templates.get("multiply", "")
            elif "factorial" in prompt_lower:
                return self.templates.get("factorial", "")
            elif "fibonacci" in prompt_lower:
                return self.templates.get("fibonacci", "")
        
        if "class" in prompt_lower:
            if "stack" in prompt_lower:
                return self.templates.get("stack", "")
            elif "queue" in prompt_lower:
                return self.templates.get("queue", "")
        
        if "sort" in prompt_lower:
            return self.templates.get("bubble sort", "")
        
        if "search" in prompt_lower:
            return self.templates.get("binary search", "")
        
        # Default response
        return f"""# {prompt}
def solution():
    # TODO: Implement solution
    pass"""
    
    def _add_vortex_comment(self, code, prompt):
        """Add vortex generation comment"""
        return f"# Generated by Vortex Code for: {prompt}\n{code}"


# For compatibility with the original module
VortexCodeGenerator = SimpleVortexGenerator