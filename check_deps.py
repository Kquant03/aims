# Save as check_deps.py
import subprocess
import sys

# List of imports from your project
required_packages = {
    'aiofiles': 'aiofiles',
    'aioboto3': 'aioboto3', 
    'mem0ai': 'mem0ai',
    'redis': 'redis',
    'psycopg2': 'psycopg2-binary',
    'pgvector': 'pgvector',
    'qdrant_client': 'qdrant-client',
    'neo4j': 'neo4j',
    'websockets': 'websockets',
    'yaml': 'pyyaml',
    'dotenv': 'python-dotenv',
    'anthropic': 'anthropic',
    'openai': 'openai',
    'aiohttp': 'aiohttp',
    'aiohttp_cors': 'aiohttp-cors',
    'aiohttp_session': 'aiohttp-session',
    'cryptography': 'cryptography',
    'transformers': 'transformers',
    'torch': 'torch',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn'
}

missing = []

for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"✓ {module}")
    except ImportError:
        print(f"✗ {module} (missing)")
        missing.append(package)

if missing:
    print(f"\nInstalling missing packages: {', '.join(missing)}")
    subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
else:
    print("\nAll dependencies are installed!")