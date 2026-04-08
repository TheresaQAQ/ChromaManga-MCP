#!/usr/bin/env python3
"""
ChromaManga MCP Setup Checker
Diagnose MCP server configuration and dependencies
"""

import os
import sys
import json
from pathlib import Path

def check_python():
    """Check Python version"""
    print("1. Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("   ✓ Python version OK")
        return True
    else:
        print("   ✗ Python 3.8+ required")
        return False

def check_mcp_sdk():
    """Check if MCP SDK is installed"""
    print("\n2. Checking MCP SDK...")
    try:
        import mcp
        print(f"   ✓ MCP SDK installed (version: {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown'})")
        return True
    except ImportError:
        print("   ✗ MCP SDK not installed")
        print("   Run: pip install mcp")
        return False

def check_dependencies():
    """Check ChromaManga dependencies"""
    print("\n3. Checking ChromaManga dependencies...")
    required = [
        "torch",
        "diffusers",
        "transformers",
        "PIL",
        "cv2",
        "numpy",
        "ultralytics"
    ]
    
    missing = []
    for pkg in required:
        try:
            if pkg == "PIL":
                __import__("PIL")
            elif pkg == "cv2":
                __import__("cv2")
            else:
                __import__(pkg)
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ✗ {pkg} not found")
            missing.append(pkg)
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    return True

def check_config_file():
    """Check if config.py exists and is valid"""
    print("\n4. Checking config.py...")
    if not os.path.exists("config.py"):
        print("   ✗ config.py not found")
        return False
    
    try:
        import config
        print("   ✓ config.py loaded")
        
        # Check critical paths
        if hasattr(config, 'models_dir'):
            print(f"   Models dir: {config.models_dir}")
            if os.path.exists(config.models_dir):
                print("   ✓ Models directory exists")
            else:
                print("   ⚠ Models directory not found")
        
        return True
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")
        return False

def check_mcp_config():
    """Check Kiro MCP configuration"""
    print("\n5. Checking Kiro MCP configuration...")
    
    config_path = Path.home() / ".kiro" / "settings" / "mcp.json"
    
    if not config_path.exists():
        print(f"   ✗ MCP config not found: {config_path}")
        print("   Create it with the provided example")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"   ✓ MCP config found: {config_path}")
        
        if "mcpServers" in config and "chromamanga" in config["mcpServers"]:
            server_config = config["mcpServers"]["chromamanga"]
            print("   ✓ chromamanga server configured")
            
            if "command" in server_config:
                print(f"   Command: {server_config['command']}")
            
            if "args" in server_config and len(server_config["args"]) > 0:
                script_path = server_config["args"][0]
                print(f"   Script: {script_path}")
                
                if os.path.exists(script_path):
                    print("   ✓ MCP server script exists")
                else:
                    print("   ✗ MCP server script not found")
                    return False
            
            if server_config.get("disabled", False):
                print("   ⚠ Server is disabled")
            else:
                print("   ✓ Server is enabled")
            
            return True
        else:
            print("   ✗ chromamanga server not configured")
            return False
            
    except Exception as e:
        print(f"   ✗ Error reading config: {e}")
        return False

def check_mcp_server_script():
    """Check if MCP server script exists"""
    print("\n6. Checking MCP server script...")
    
    if not os.path.exists("chromamanga_mcp_server.py"):
        print("   ✗ chromamanga_mcp_server.py not found")
        return False
    
    print("   ✓ chromamanga_mcp_server.py exists")
    
    if not os.path.exists("task_manager.py"):
        print("   ✗ task_manager.py not found")
        return False
    
    print("   ✓ task_manager.py exists")
    return True

def main():
    print("=" * 60)
    print("ChromaManga MCP Setup Checker")
    print("=" * 60)
    
    checks = [
        check_python(),
        check_mcp_sdk(),
        check_dependencies(),
        check_config_file(),
        check_mcp_server_script(),
        check_mcp_config(),
    ]
    
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print("\nYou can now:")
        print("1. Restart Kiro to load the MCP server")
        print("2. Test in Kiro Chat: '请使用 get_config 工具'")
        print("3. Or run manually: python chromamanga_mcp_server.py")
    else:
        print(f"✗ {total - passed} check(s) failed ({passed}/{total} passed)")
        print("\nPlease fix the issues above before using MCP server")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
