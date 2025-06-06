#!/bin/bash
# quickfix_and_run.sh - Quick fix to get AIMS running NOW

echo "🚀 AIMS Quick Fix & Run"
echo "======================="

# Fix imports
echo "🔧 Fixing imports..."
chmod +x fix_imports.sh 2>/dev/null
./fix_imports.sh 2>/dev/null || {
    # If fix_imports.sh doesn't exist, do it inline
    find src -name "*.py" -type f -exec sed -i 's/from core\./from src.core./g' {} \; 2>/dev/null
    find src -name "*.py" -type f -exec sed -i 's/from api\./from src.api./g' {} \; 2>/dev/null
    find src -name "*.py" -type f -exec sed -i 's/from utils\./from src.utils./g' {} \; 2>/dev/null
}

# Create run_aims.py if it doesn't exist
if [ ! -f run_aims.py ]; then
    echo "📝 Creating launcher..."
    cat > run_aims.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simple test to see if we can start
print("\n🧠 AIMS Starting...\n")

try:
    # Try the full version
    from src.main import main
    import asyncio
    asyncio.run(main())
except Exception as e:
    print(f"Error: {e}")
    print("\nRunning minimal version...\n")
    
    # Minimal version
    print("AIMS Minimal Mode")
    print("=" * 50)
    print(f"API Key: {'✓ Set' if os.environ.get('ANTHROPIC_API_KEY') else '✗ Not Set'}")
    print("=" * 50)
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() == 'quit':
                break
            print(f"Echo: {user_input}\n")
        except KeyboardInterrupt:
            break
    
    print("\n👋 Goodbye!")
EOF
fi

# Make it executable
chmod +x run_aims.py

# Run it!
echo ""
echo "🚀 Starting AIMS..."
echo ""
python3 run_aims.py