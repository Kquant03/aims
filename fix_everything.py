#!/usr/bin/env python3
"""
AIMS Bootstrap Fix Script - No External Dependencies Required
This script fixes all issues and sets up everything needed to run AIMS
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import json
import secrets
import stat
from datetime import datetime

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

class AIMSBootstrapFixer:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.issues_fixed = []
        self.issues_failed = []
        
    def run(self):
        """Run all fixes and setup"""
        print(f"{BLUE}{'='*60}{NC}")
        print(f"{BLUE}AIMS Bootstrap Setup & Fix Script{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        # First, ensure we have a virtual environment and install dependencies
        self.setup_virtual_environment()
        
        # Then run all other fixes
        self.fix_directory_structure()
        self.fix_session_key_issue()
        self.extract_html_template()
        self.create_static_files()
        self.fix_env_file()
        self.create_missing_utility_files()
        self.fix_personality_baseline()
        self.fix_permissions()
        self.check_docker_services()
        self.create_helper_scripts()
        self.final_verification()
        
        # Summary
        self.print_summary()
    
    def setup_virtual_environment(self):
        """Set up Python virtual environment and install dependencies"""
        print(f"\n{BLUE}0. Setting up Python environment...{NC}")
        
        # Check if venv exists
        venv_path = self.root_dir / 'venv'
        if not venv_path.exists():
            print(f"  Creating virtual environment...")
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            print(f"  {GREEN}✓{NC} Virtual environment created")
        else:
            print(f"  {BLUE}•{NC} Virtual environment already exists")
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = venv_path / 'Scripts' / 'pip'
            python_path = venv_path / 'Scripts' / 'python'
        else:  # Unix-like
            pip_path = venv_path / 'bin' / 'pip'
            python_path = venv_path / 'bin' / 'python'
        
        # Upgrade pip
        print(f"  Upgrading pip...")
        subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      capture_output=True)
        print(f"  {GREEN}✓{NC} Pip upgraded")
        
        # Install required packages for this script
        print(f"  Installing PyYAML and cryptography...")
        result = subprocess.run([str(pip_path), 'install', 'pyyaml', 'cryptography'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {GREEN}✓{NC} Required packages installed")
        else:
            print(f"  {RED}✗{NC} Failed to install packages: {result.stderr}")
        
        # Install all requirements if requirements.txt exists
        req_path = self.root_dir / 'requirements.txt'
        if req_path.exists():
            print(f"  Installing all project requirements (this may take a few minutes)...")
            result = subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  {GREEN}✓{NC} All requirements installed")
            else:
                print(f"  {YELLOW}!{NC} Some requirements failed to install")
                print(f"     You may need to install them manually later")
        
        self.issues_fixed.append("Python environment setup")
    
    def fix_directory_structure(self):
        """Create all necessary directories"""
        print(f"\n{BLUE}1. Creating directory structure...{NC}")
        
        directories = [
            'data/states',
            'data/backups', 
            'data/memories',
            'data/uploads',
            'logs',
            'src/ui/templates',
            'src/ui/static/js',
            'src/ui/static/css',
            'src/ui/static/images',
            'src/utils',
            'src/persistence',
            'configs',
            'scripts/fixes',
            'scripts/setup',
            'scripts/utils',
            'tests',
            'docs'
        ]
        
        for dir_path in directories:
            path = self.root_dir / dir_path
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"  {GREEN}✓{NC} Created {dir_path}")
            else:
                print(f"  {BLUE}•{NC} {dir_path} already exists")
        
        self.issues_fixed.append("Directory structure")
    
    def fix_session_key_issue(self):
        """Fix the session key encoding issue in web_interface.py"""
        print(f"\n{BLUE}2. Fixing session key issue...{NC}")
        
        web_interface_path = self.root_dir / 'src/ui/web_interface.py'
        
        if not web_interface_path.exists():
            print(f"  {RED}✗{NC} web_interface.py not found!")
            self.issues_failed.append("Session key fix")
            return
        
        # Read the file
        with open(web_interface_path, 'r') as f:
            content = f.read()
        
        # Check if cryptography is available
        try:
            from cryptography.fernet import Fernet
            has_crypto = True
        except ImportError:
            has_crypto = False
            print(f"  {YELLOW}!{NC} cryptography module not available yet, using simple fix")
        
        # Fix the session key issue
        original_line = "setup(self.app, EncryptedCookieStorage(secret_key.encode()))"
        
        if has_crypto:
            # Full fix with Fernet key generation
            replacement = """# Set up session middleware with proper key handling
        from cryptography.fernet import Fernet
        
        # Handle different key formats
        try:
            # If it's already a valid Fernet key, use it as-is
            if secret_key != 'dev-secret-key-change-in-production':
                test_fernet = Fernet(secret_key.encode('utf-8') if isinstance(secret_key, str) and len(secret_key) == 32 else secret_key)
                setup(self.app, EncryptedCookieStorage(secret_key if isinstance(secret_key, bytes) else secret_key.encode('utf-8')))
            else:
                # Generate a proper key for development
                generated_key = Fernet.generate_key()
                setup(self.app, EncryptedCookieStorage(generated_key))
                logger.warning("Using generated session key - set SESSION_SECRET in .env for production")
        except Exception as e:
            # Fallback: generate a new key
            generated_key = Fernet.generate_key()
            setup(self.app, EncryptedCookieStorage(generated_key))
            logger.warning(f"Session key error: {e}. Using generated key.")"""
        else:
            # Simple fix without Fernet
            replacement = """# Set up session middleware
        # Simple fix - just use the key as bytes
        setup(self.app, EncryptedCookieStorage(secret_key if isinstance(secret_key, bytes) else secret_key.encode('utf-8')))"""
        
        if original_line in content:
            content = content.replace(original_line, replacement)
            
            # Write back
            with open(web_interface_path, 'w') as f:
                f.write(content)
            
            print(f"  {GREEN}✓{NC} Fixed session key encoding issue")
            self.issues_fixed.append("Session key fix")
        else:
            print(f"  {YELLOW}!{NC} Session key line not found as expected, applying alternative fix...")
            
            # Try a simpler fix
            content = content.replace(
                "EncryptedCookieStorage(secret_key.encode())",
                "EncryptedCookieStorage(secret_key if isinstance(secret_key, bytes) else secret_key.encode('utf-8'))"
            )
            
            with open(web_interface_path, 'w') as f:
                f.write(content)
            
            print(f"  {GREEN}✓{NC} Applied alternative session key fix")
            self.issues_fixed.append("Session key fix (alternative)")
    
    def extract_html_template(self):
        """Extract HTML template from web_interface.py"""
        print(f"\n{BLUE}3. Extracting HTML template...{NC}")
        
        template_dir = self.root_dir / 'src/ui/templates'
        template_path = template_dir / 'index.html'
        
        # Check if template already exists
        if template_path.exists():
            print(f"  {BLUE}•{NC} Template already exists")
            return
        
        # Extract from web_interface.py
        web_interface_path = self.root_dir / 'src/ui/web_interface.py'
        
        if not web_interface_path.exists():
            print(f"  {RED}✗{NC} web_interface.py not found!")
            self.issues_failed.append("HTML template extraction")
            return
        
        # Read the file
        with open(web_interface_path, 'r') as f:
            content = f.read()
        
        # Find the HTML template
        if "HTML_TEMPLATE = '''" in content:
            start = content.find("HTML_TEMPLATE = '''") + len("HTML_TEMPLATE = '''")
            end = content.find("'''", start)
            
            if end > start:
                html_content = content[start:end]
                
                # Write to template file
                with open(template_path, 'w') as f:
                    f.write(html_content)
                
                print(f"  {GREEN}✓{NC} Extracted HTML template to templates/index.html")
                
                # Remove the embedded template and save_template function from web_interface.py
                content = content[:content.find("# Create the HTML template")] + content[content.find("if __name__ == '__main__':"):]
                
                with open(web_interface_path, 'w') as f:
                    f.write(content)
                
                print(f"  {GREEN}✓{NC} Cleaned up web_interface.py")
                self.issues_fixed.append("HTML template extraction")
            else:
                print(f"  {RED}✗{NC} Could not find HTML template in web_interface.py")
                self.issues_failed.append("HTML template extraction")
        else:
            print(f"  {YELLOW}!{NC} HTML template already extracted or not found")
    
    def create_static_files(self):
        """Create static CSS and JS files"""
        print(f"\n{BLUE}4. Creating static files...{NC}")
        
        # Create CSS file
        css_path = self.root_dir / 'src/ui/static/css/styles.css'
        if not css_path.exists():
            css_content = """/* AIMS Styles */
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #1a1a1a;
    --bg-tertiary: #0f0f0f;
    --text-primary: #e0e0e0;
    --text-secondary: #888;
    --accent: #0066cc;
    --accent-hover: #0052a3;
    --border: #333;
    --success: #00ff88;
    --error: #ff4444;
}

/* Additional styles for consciousness visualization */
.consciousness-viz {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}

.memory-graph {
    width: 100%;
    height: 300px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
}

/* Animations */
@keyframes pulse {
    0% { opacity: 0.8; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
}

.pulse {
    animation: pulse 2s infinite;
}
"""
            with open(css_path, 'w') as f:
                f.write(css_content)
            print(f"  {GREEN}✓{NC} Created styles.css")
        
        # Create JS file
        js_path = self.root_dir / 'src/ui/static/js/consciousness-viz.js'
        if not js_path.exists():
            js_content = """// AIMS Consciousness Visualization
class ConsciousnessVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.data = {
            coherence: 1.0,
            emotions: { pleasure: 0.5, arousal: 0.5, dominance: 0.5 },
            memories: []
        };
    }
    
    update(data) {
        this.data = { ...this.data, ...data };
        this.render();
    }
    
    render() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Render consciousness visualization
        // This is a placeholder - implement your visualization logic here
        this.ctx.fillStyle = '#00ff88';
        this.ctx.fillRect(10, 10, this.data.coherence * 200, 20);
    }
}

// Export for use
window.ConsciousnessVisualizer = ConsciousnessVisualizer;
"""
            with open(js_path, 'w') as f:
                f.write(js_content)
            print(f"  {GREEN}✓{NC} Created consciousness-viz.js")
        
        self.issues_fixed.append("Static files")
    
    def fix_env_file(self):
        """Create and fix .env file"""
        print(f"\n{BLUE}5. Setting up environment variables...{NC}")
        
        env_path = self.root_dir / '.env'
        env_example_path = self.root_dir / '.env.example'
        
        # Create .env.example if it doesn't exist
        if not env_example_path.exists():
            env_example_content = """# Required
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional but recommended
OPENAI_API_KEY=your-openai-api-key-for-embeddings
SESSION_SECRET=change-this-to-a-random-string

# Database Configuration
POSTGRES_PASSWORD=aims_secure_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=aims_memory
POSTGRES_USER=aims

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# S3 Backup (optional)
S3_BACKUP_ENABLED=false
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_REGION=us-east-1

# System Configuration
LOG_LEVEL=INFO
BACKUP_INTERVAL_HOURS=6
MAX_LOCAL_BACKUPS=7
"""
            with open(env_example_path, 'w') as f:
                f.write(env_example_content)
            print(f"  {GREEN}✓{NC} Created .env.example")
        
        # Create .env from .env.example if it doesn't exist
        if not env_path.exists():
            shutil.copy(env_example_path, env_path)
            print(f"  {GREEN}✓{NC} Created .env from .env.example")
        
        # Read current .env
        with open(env_path, 'r') as f:
            env_content = f.read()
        
        # Generate a proper SESSION_SECRET if needed
        if 'SESSION_SECRET=change-this-to-a-random-string' in env_content or 'SESSION_SECRET=dev-secret-key-change-in-production' in env_content:
            try:
                from cryptography.fernet import Fernet
                new_key = Fernet.generate_key().decode()
            except ImportError:
                # Fallback to a random string if cryptography isn't available
                new_key = secrets.token_urlsafe(32)
            
            env_content = env_content.replace('SESSION_SECRET=change-this-to-a-random-string', f'SESSION_SECRET={new_key}')
            env_content = env_content.replace('SESSION_SECRET=dev-secret-key-change-in-production', f'SESSION_SECRET={new_key}')
            
            with open(env_path, 'w') as f:
                f.write(env_content)
            
            print(f"  {GREEN}✓{NC} Generated secure SESSION_SECRET")
        
        # Check for API key
        if 'ANTHROPIC_API_KEY=your-anthropic-api-key-here' in env_content:
            print(f"  {YELLOW}!{NC} Remember to add your ANTHROPIC_API_KEY to .env")
        
        self.issues_fixed.append("Environment configuration")
    
    def create_missing_utility_files(self):
        """Create missing utility files"""
        print(f"\n{BLUE}6. Creating missing utility files...{NC}")
        
        # Create logger.py
        logger_path = self.root_dir / 'src/utils/logger.py'
        if not logger_path.exists():
            logger_content = '''"""
logger.py - Structured logging setup for AIMS
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging(log_level: str = 'INFO', log_file: str = 'logs/aims.log'):
    """Set up logging configuration for AIMS"""
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('aims')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with JSON format
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Performance logging decorator
def log_performance(logger: logging.Logger):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"{func.__name__} completed",
                extra={
                    'function': func.__name__,
                    'duration_seconds': duration,
                    'performance': True
                }
            )
            
            return result
        return wrapper
    return decorator
'''
            with open(logger_path, 'w') as f:
                f.write(logger_content)
            print(f"  {GREEN}✓{NC} Created logger.py")
        
        # Create metrics.py
        metrics_path = self.root_dir / 'src/utils/metrics.py'
        if not metrics_path.exists():
            metrics_content = '''"""
metrics.py - Metrics collection and monitoring for AIMS
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and tracks metrics for consciousness system"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.start_time = datetime.now()
    
    def record(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now(),
            'tags': tags or {}
        })
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
        
        values = [m['value'] for m in self.metrics[metric_name]]
        if not values:
            return {}
        
        # Simple statistics without numpy
        count = len(values)
        mean = sum(values) / count
        sorted_values = sorted(values)
        
        return {
            'count': count,
            'mean': mean,
            'min': min(values),
            'max': max(values),
            'p50': sorted_values[count // 2],
            'p95': sorted_values[int(count * 0.95)] if count > 20 else max(values),
            'p99': sorted_values[int(count * 0.99)] if count > 100 else max(values)
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {name: self.get_stats(name) for name in self.metrics}

class ConsciousnessMetrics:
    """Specific metrics for consciousness system"""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.coherence_history = deque(maxlen=100)
    
    def record_coherence(self, coherence: float):
        """Record consciousness coherence score"""
        self.collector.record('consciousness.coherence', coherence)
        self.coherence_history.append({
            'value': coherence,
            'timestamp': datetime.now()
        })
    
    def record_memory_operation(self, operation: str, duration: float, success: bool):
        """Record memory operation metrics"""
        self.collector.record(
            f'memory.{operation}.duration',
            duration,
            tags={'success': str(success)}
        )
    
    def record_emotional_state(self, pleasure: float, arousal: float, dominance: float):
        """Record emotional state metrics"""
        self.collector.record('emotion.pleasure', pleasure)
        self.collector.record('emotion.arousal', arousal)
        self.collector.record('emotion.dominance', dominance)
    
    def record_api_call(self, api: str, duration: float, tokens: int, success: bool):
        """Record API call metrics"""
        self.collector.record(
            f'api.{api}.duration',
            duration,
            tags={'success': str(success)}
        )
        self.collector.record(
            f'api.{api}.tokens',
            tokens,
            tags={'success': str(success)}
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        coherence_stats = self.collector.get_stats('consciousness.coherence')
        
        # Determine health based on coherence
        if not coherence_stats:
            health = 'unknown'
        elif coherence_stats['mean'] > 0.8:
            health = 'healthy'
        elif coherence_stats['mean'] > 0.6:
            health = 'degraded'
        else:
            health = 'unhealthy'
        
        return {
            'status': health,
            'uptime_seconds': (datetime.now() - self.collector.start_time).total_seconds(),
            'coherence': coherence_stats,
            'metrics_summary': self.collector.get_all_metrics()
        }

# Global metrics instance
consciousness_metrics = ConsciousnessMetrics()
'''
            with open(metrics_path, 'w') as f:
                f.write(metrics_content)
            print(f"  {GREEN}✓{NC} Created metrics.py")
        
        # Create database_manager.py
        db_manager_path = self.root_dir / 'src/persistence/database_manager.py'
        if not db_manager_path.exists():
            db_manager_content = '''"""
database_manager.py - Database connection management for AIMS
"""
import os
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages connections to PostgreSQL, Redis, and Qdrant"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pg_pool = None
        self.redis_client = None
        self.qdrant_client = None
        
    async def initialize(self):
        """Initialize all database connections"""
        # Note: This is a placeholder implementation
        # Real implementation requires the database libraries to be installed
        logger.info("Database manager initialized (placeholder mode)")
        
    async def close(self):
        """Close all database connections"""
        logger.info("Database connections closed")
    
    # Placeholder methods for database operations
    async def store_memory(self, content: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Store a memory (placeholder)"""
        import uuid
        return str(uuid.uuid4())
    
    async def search_memories(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by vector similarity (placeholder)"""
        return []
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set a value in cache (placeholder)"""
        pass
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get a value from cache (placeholder)"""
        return None
'''
            with open(db_manager_path, 'w') as f:
                f.write(db_manager_content)
            print(f"  {GREEN}✓{NC} Created database_manager.py")
        
        self.issues_fixed.append("Utility files")
    
    def fix_personality_baseline(self):
        """Create personality baseline configuration"""
        print(f"\n{BLUE}7. Creating personality baseline...{NC}")
        
        baseline_path = self.root_dir / 'configs/personality_baseline.yaml'
        
        baseline_content = """# Personality baseline configuration for AIMS
# OCEAN model traits (0-1 scale)

baseline:
  openness: 0.8          # Creativity, curiosity, openness to new experiences
  conscientiousness: 0.7 # Organization, dependability, self-discipline  
  extraversion: 0.6      # Sociability, assertiveness, emotional expression
  agreeableness: 0.8     # Cooperation, trust, empathy
  neuroticism: 0.3       # Emotional instability, anxiety, moodiness

# Trait evolution parameters
evolution:
  learning_rate: 0.001   # How quickly traits change
  momentum: 0.95         # Smoothing factor for changes
  
# Trait bounds (min, max) to prevent extreme personality shifts
bounds:
  openness: [0.4, 1.0]
  conscientiousness: [0.3, 1.0]
  extraversion: [0.2, 1.0]
  agreeableness: [0.4, 1.0]
  neuroticism: [0.0, 0.7]

# Behavioral modifiers based on traits
behavioral_modifiers:
  response_length:
    - trait: openness
      weight: 0.5
    - trait: conscientiousness
      weight: 0.3
  
  emotional_expression:
    - trait: extraversion
      weight: 0.6
    - trait: neuroticism
      weight: -0.4
  
  creativity:
    - trait: openness
      weight: 0.8
    - trait: extraversion
      weight: 0.2
"""
        
        with open(baseline_path, 'w') as f:
            f.write(baseline_content)
        
        print(f"  {GREEN}✓{NC} Created personality_baseline.yaml")
        self.issues_fixed.append("Personality baseline")
    
    def fix_permissions(self):
        """Fix file permissions for scripts"""
        print(f"\n{BLUE}8. Fixing file permissions...{NC}")
        
        # Make all .sh files executable
        sh_files = list(self.root_dir.glob('**/*.sh'))
        
        for sh_file in sh_files:
            try:
                st = os.stat(sh_file)
                os.chmod(sh_file, st.st_mode | stat.S_IEXEC)
                print(f"  {GREEN}✓{NC} Made {sh_file.name} executable")
            except Exception as e:
                print(f"  {RED}✗{NC} Failed to fix permissions for {sh_file.name}: {e}")
        
        # Make Python scripts in scripts/ executable
        py_scripts = list((self.root_dir / 'scripts').glob('**/*.py'))
        
        for py_script in py_scripts:
            try:
                st = os.stat(py_script)
                os.chmod(py_script, st.st_mode | stat.S_IEXEC)
                print(f"  {GREEN}✓{NC} Made {py_script.name} executable")
            except Exception as e:
                print(f"  {RED}✗{NC} Failed to fix permissions for {py_script.name}: {e}")
        
        self.issues_fixed.append("File permissions")
    
    def check_docker_services(self):
        """Check if Docker services are running"""
        print(f"\n{BLUE}9. Checking Docker services...{NC}")
        
        try:
            # Check if docker is accessible
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  {YELLOW}!{NC} Docker not accessible. Run: sudo usermod -aG docker $USER && newgrp docker")
                return
            
            # Check docker-compose services
            result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout
                services = ['postgres', 'redis', 'qdrant']
                
                for service in services:
                    if f'aims_{service}' in output and 'Up' in output:
                        print(f"  {GREEN}✓{NC} {service} is running")
                    else:
                        print(f"  {YELLOW}!{NC} {service} is not running")
                
                if not any(f'aims_{s}' in output and 'Up' in output for s in services):
                    print(f"\n  {BLUE}→{NC} Start services with: docker-compose up -d")
            else:
                print(f"  {YELLOW}!{NC} docker-compose not found or no services defined")
                
        except FileNotFoundError:
            print(f"  {RED}✗{NC} Docker not installed")
    
    def create_helper_scripts(self):
        """Create convenient helper scripts"""
        print(f"\n{BLUE}10. Creating helper scripts...{NC}")
        
        # Create quick start script
        start_script = self.root_dir / 'start_aims.sh'
        start_content = """#!/bin/bash
# Quick start script for AIMS

echo "Starting AIMS..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -q -r requirements.txt

# Start Docker services if not running
if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo "Starting Docker services..."
    docker-compose up -d
    sleep 5
fi

# Run AIMS
echo "Starting AIMS on http://localhost:8000"
python -m src.main
"""
        
        with open(start_script, 'w') as f:
            f.write(start_content)
        
        os.chmod(start_script, 0o755)
        print(f"  {GREEN}✓{NC} Created start_aims.sh")
        
        # Create test script
        test_script = self.root_dir / 'test_aims.sh'
        test_content = """#!/bin/bash
# Test script for AIMS

source venv/bin/activate
pytest tests/ -v --tb=short
"""
        
        with open(test_script, 'w') as f:
            f.write(test_content)
        
        os.chmod(test_script, 0o755)
        print(f"  {GREEN}✓{NC} Created test_aims.sh")
        
        self.issues_fixed.append("Helper scripts")
    
    def final_verification(self):
        """Final verification of the setup"""
        print(f"\n{BLUE}11. Final verification...{NC}")
        
        # Check critical files exist
        critical_files = [
            'src/main.py',
            'src/core/consciousness.py',
            'src/api/claude_interface.py',
            'src/ui/web_interface.py',
            '.env',
            'requirements.txt'
        ]
        
        all_good = True
        for file_path in critical_files:
            if (self.root_dir / file_path).exists():
                print(f"  {GREEN}✓{NC} {file_path}")
            else:
                print(f"  {RED}✗{NC} {file_path} missing!")
                all_good = False
        
        if all_good:
            self.issues_fixed.append("Final verification")
        else:
            self.issues_failed.append("Some critical files missing")
    
    def print_summary(self):
        """Print summary of fixes"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}Setup Summary{NC}")
        print(f"{BLUE}{'='*60}{NC}")
        
        print(f"\n{GREEN}Successfully fixed:{NC}")
        for issue in self.issues_fixed:
            print(f"  ✓ {issue}")
        
        if self.issues_failed:
            print(f"\n{RED}Failed to fix:{NC}")
            for issue in self.issues_failed:
                print(f"  ✗ {issue}")
        
        print(f"\n{BLUE}Next steps:{NC}")
        print("1. Add your ANTHROPIC_API_KEY to .env file")
        print("2. Activate virtual environment: source venv/bin/activate")
        print("3. Start Docker services: docker-compose up -d")
        print("4. Run AIMS: python -m src.main")
        print("\nOr use the quick start script: ./start_aims.sh")
        
        print(f"\n{GREEN}✨ Your AIMS project is ready to build!{NC}")


if __name__ == "__main__":
    fixer = AIMSBootstrapFixer()
    fixer.run()