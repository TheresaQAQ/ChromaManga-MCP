#!/usr/bin/env python3
"""
Quick test script for ChromaManga MCP Server
Tests basic functionality without full MCP client
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp_server.chromamanga_mcp_server import (
    initialize_models,
    handle_create_task,
    handle_get_task_status,
    handle_extract_lineart,
    handle_get_config,
    task_manager
)


async def test_basic_flow():
    """Test basic MCP server functionality"""
    print("=" * 60)
    print("ChromaManga MCP Server Test")
    print("=" * 60)
    
    # Initialize
    print("\n1. Initializing models...")
    try:
        await initialize_models()
        print("✓ Models loaded")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        return
    
    # Test get_config
    print("\n2. Testing get_config...")
    try:
        result = await handle_get_config({})
        print(result[0].text)
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test create_task
    print("\n3. Testing create_task...")
    test_image = "inputs/553d7fe3f77db5f73264cad52de45117.png"
    
    if not os.path.exists(test_image):
        print(f"✗ Test image not found: {test_image}")
        print("Please provide a valid image path")
        return
    
    try:
        result = await handle_create_task({
            "image_path": test_image,
            "task_name": "test_task"
        })
        print(result[0].text)
        
        # Extract task_id from response
        task_id = None
        for line in result[0].text.split('\n'):
            if line.startswith("Task ID:"):
                task_id = line.split(": ")[1].strip()
                break
        
        if not task_id:
            print("✗ Failed to get task_id")
            return
        
        print(f"\n✓ Task created: {task_id}")
        
        # Test get_task_status
        print("\n4. Testing get_task_status...")
        result = await handle_get_task_status({"task_id": task_id})
        print(result[0].text)
        
        # Test extract_lineart
        print("\n5. Testing extract_lineart...")
        result = await handle_extract_lineart({
            "task_id": task_id,
            "method": "lineart_anime"
        })
        print(result[0].text)
        
        print("\n" + "=" * 60)
        print("✓ Basic tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    asyncio.run(test_basic_flow())
