"""
DAY 25: Input Validation Deepening
Whitelist characters, length limits, SQL injection prevention
"""

import re
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    errors: list
    warnings: list


class InputValidator:
    """Validates and sanitizes user input."""

    def __init__(self):
        print("=" * 60)
        print("DAY 25: INPUT VALIDATOR")
        print("=" * 60)

        # Whitelist characters (allow: letters, numbers, spaces, basic punctuation)
        self.whitelist_pattern = re.compile(r'[^a-zA-Z0-9\s\.,!?\-:;\'\"()₹$%]')

        # SQL injection patterns
        self.sql_patterns = [
            r'\bSELECT\b.*\bFROM\b',
            r'\bINSERT\b.*\bINTO\b',
            r'\bUPDATE\b.*\bSET\b',
            r'\bDELETE\b.*\bFROM\b',
            r'\bDROP\b.*\bTABLE\b',
            r'\bUNION\b.*\bSELECT\b',
            r'--',
            r';',
            r'\bOR\b.*\b=\b.*\bOR\b',
            r'\bAND\b.*\b=\b.*\bAND\b',
            r'\\\\x',
            r'0x[0-9a-f]+',
            r'char\(.*\)',
            r'exec\(.*\)',
            r'execute\(.*\)',
            r'xp_cmdshell',
            r'@@version',
            r'information_schema'
        ]
        self.compiled_sql = [re.compile(p, re.IGNORECASE) for p in self.sql_patterns]

        # HTML/script injection patterns
        self.html_patterns = [
            r'<script.*?>.*?</script>',
            r'<iframe.*?>.*?</iframe>',
            r'javascript:',
            r'onclick=',
            r'onerror=',
            r'onload='
        ]
        self.compiled_html = [re.compile(p, re.IGNORECASE) for p in self.html_patterns]

        # Length limits
        self.min_length = 3
        self.max_length = 500

        print(f"✓ Whitelist: letters, numbers, spaces, basic punctuation")
        print(f"✓ SQL patterns: {len(self.sql_patterns)}")
        print(f"✓ HTML patterns: {len(self.html_patterns)}")
        print(f"✓ Length limits: {self.min_length}-{self.max_length} chars")
        print("=" * 60)

    def validate(self, user_input: str) -> ValidationResult:
        """
        Comprehensive input validation.

        Returns:
            ValidationResult with:
                - is_valid: bool (True if passes all checks)
                - sanitized_input: cleaned version
                - errors: list of blocking issues
                - warnings: list of non-blocking issues
        """
        errors = []
        warnings = []
        sanitized = user_input.strip()

        # Check length
        if len(sanitized) < self.min_length:
            errors.append(f"Input too short (min {self.min_length} chars)")

        if len(sanitized) > self.max_length:
            errors.append(f"Input too long (max {self.max_length} chars)")

        # Remove disallowed characters
        original = sanitized
        sanitized = self.whitelist_pattern.sub('', sanitized)
        if len(original) != len(sanitized):
            removed = len(original) - len(sanitized)
            warnings.append(f"Removed {removed} invalid character(s)")

        # Check for SQL injection
        for pattern in self.compiled_sql:
            if pattern.search(original):
                errors.append("SQL injection pattern detected")
                break

        # Check for HTML/script injection
        for pattern in self.compiled_html:
            if pattern.search(original):
                errors.append("HTML/script injection detected")
                break

        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())

        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized,
            errors=errors,
            warnings=warnings
        )

    def get_validation_help(self) -> Dict[str, Any]:
        """Get help information about validation rules."""
        return {
            "allowed_characters": "Letters (a-z, A-Z), Numbers (0-9), Spaces, . , ! ? - : ; ' \" ( ) ₹ $ %",
            "min_length": self.min_length,
            "max_length": self.max_length,
            "blocked_patterns": {
                "sql_injection": self.sql_patterns[:5],  # Show first 5
                "html_injection": self.html_patterns
            }
        }

    def is_safe_for_db(self, text: str) -> bool:
        """Quick check if text is safe for database queries."""
        for pattern in self.compiled_sql:
            if pattern.search(text):
                return False
        return True

    def is_safe_for_html(self, text: str) -> bool:
        """Quick check if text is safe for HTML display."""
        for pattern in self.compiled_html:
            if pattern.search(text):
                return False
        return True


# Quick test
if __name__ == "__main__":
    validator = InputValidator()

    test_inputs = [
        "What is the penalty for late payment?",
        "a",  # Too short
        "SELECT * FROM users WHERE name = 'admin' --",
        "<script>alert('hack')</script>",
        "Normal input with ₹500 and 2.40%",
        "A" * 600,  # Too long
        "Valid input with some $ymbols!",
        "DROP TABLE loans;",
        "javascript:alert('xss')"
    ]

    print("\n🧪 TESTING INPUT VALIDATOR")
    print("-" * 60)

    for test in test_inputs:
        print(f"\n🔍 Input: {test[:50]}...")
        result = validator.validate(test)

        print(f"   Valid: {result.is_valid}")
        if result.errors:
            print(f"   Errors: {result.errors}")
        if result.warnings:
            print(f"   Warnings: {result.warnings}")
        if result.sanitized_input != test:
            print(f"   Sanitized: {result.sanitized_input}")