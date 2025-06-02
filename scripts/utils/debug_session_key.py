#!/usr/bin/env python3
"""
debug_session_key.py - Debug session key loading issue
"""

import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Load environment variables
load_dotenv()

print("🔍 Debugging SESSION_SECRET loading...")
print("=" * 50)

# Check if .env file exists
if os.path.exists('.env'):
    print("✅ .env file exists")
    
    # Read .env file directly
    with open('.env', 'r') as f:
        for line in f:
            if 'SESSION_SECRET' in line:
                print(f"📄 In .env file: {line.strip()}")
else:
    print("❌ .env file not found")

# Check environment variable
secret_from_env = os.environ.get('SESSION_SECRET')
print(f"\n🔧 From os.environ: {repr(secret_from_env)}")

# Check default value
secret_with_default = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')
print(f"📌 With default: {repr(secret_with_default)}")

# Test if it's a valid Fernet key
print("\n🔐 Testing Fernet key validity:")
test_keys = [
    secret_from_env,
    secret_with_default,
    'dev-secret-key-change-in-production'
]

for i, key in enumerate(test_keys):
    print(f"\nTest {i+1}: {repr(key)}")
    if key:
        try:
            # Try encoding and using it
            if isinstance(key, str):
                Fernet(key.encode())
                print("  ✅ Valid when encoded")
        except Exception as e:
            print(f"  ❌ Invalid when encoded: {e}")
            
        try:
            # Try using it directly
            if isinstance(key, str):
                Fernet(key)
                print("  ✅ Valid without encoding")
        except Exception as e:
            print(f"  ❌ Invalid without encoding: {e}")

# Generate a proper key
print("\n💡 Generating a proper Fernet key:")
proper_key = Fernet.generate_key()
print(f"Generated key (bytes): {proper_key}")
print(f"Generated key (string): {proper_key.decode()}")

# Show how to fix
print("\n🛠️  To fix the issue:")
print("1. Make sure your .env file has:")
print(f"   SESSION_SECRET={proper_key.decode()}")
print("\n2. Or modify the code to not encode the key:")
print("   Change: EncryptedCookieStorage(secret_key.encode())")
print("   To: EncryptedCookieStorage(secret_key)")