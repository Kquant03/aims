#!/usr/bin/env python3
"""
fix_session_key.py - Fix the session key handling in web_interface.py
"""

import re

def fix_session_key_handling():
    """Fix the session key handling in web_interface.py"""
    
    filepath = 'src/ui/web_interface.py'
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the line with the issue
    # Change from: setup(self.app, EncryptedCookieStorage(secret_key.encode()))
    # To: setup(self.app, EncryptedCookieStorage(secret_key.encode() if secret_key == 'dev-secret-key-change-in-production' else secret_key))
    
    # Better fix - don't encode the key at all if it's already a valid Fernet key
    old_line = "setup(self.app, EncryptedCookieStorage(secret_key.encode()))"
    
    new_code = """# Set up session middleware
        from cryptography.fernet import Fernet
        try:
            # Try to use the key as-is (for proper Fernet keys)
            if secret_key != 'dev-secret-key-change-in-production':
                Fernet(secret_key)  # Validate it's a proper Fernet key
                setup(self.app, EncryptedCookieStorage(secret_key))
            else:
                # For the default dev key, generate a proper one
                generated_key = Fernet.generate_key()
                setup(self.app, EncryptedCookieStorage(generated_key))
                logger.warning("Using generated session key - set SESSION_SECRET in .env for production")
        except Exception:
            # If validation fails, encode it (backward compatibility)
            try:
                setup(self.app, EncryptedCookieStorage(secret_key.encode()))
            except:
                # Last resort - generate a new key
                generated_key = Fernet.generate_key()
                setup(self.app, EncryptedCookieStorage(generated_key))
                logger.warning("Invalid SESSION_SECRET - using generated key")"""
    
    # Replace the problematic line
    if old_line in content:
        # Find the comment before it to maintain context
        pattern = r'# Set up session middleware\s*\n\s*' + re.escape(old_line)
        content = re.sub(pattern, new_code, content)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Fixed session key handling in web_interface.py")
        return True
    else:
        print("⚠️  Could not find the line to fix. Applying simpler fix...")
        
        # Simpler fix - just remove .encode()
        content = content.replace(
            "EncryptedCookieStorage(secret_key.encode())",
            "EncryptedCookieStorage(secret_key)"
        )
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("✅ Applied simpler fix - removed .encode()")
        return True

def main():
    print("🔧 Fixing session key handling...")
    print("=" * 50)
    
    if fix_session_key_handling():
        print("\n✅ Fix applied!")
        print("\nNow try running: python -m src.main")
    else:
        print("\n❌ Could not apply fix automatically")
        print("\nManual fix:")
        print("1. Edit src/ui/web_interface.py")
        print("2. Find: setup(self.app, EncryptedCookieStorage(secret_key.encode()))")
        print("3. Change to: setup(self.app, EncryptedCookieStorage(secret_key))")

if __name__ == "__main__":
    main()