"""
Task Manager for ChromaManga MCP Server
Handles task state, intermediate results, and workflow management
"""

import os
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from PIL import Image
import numpy as np

from . import config


class Task:
    """Represents a single colorization task"""
    
    STEPS_ORDER = [
        "lineart_extraction",
        "person_detection",
        "character_identification",
        "bubble_detection",
        "mask_generation",
        "inference",
        "postprocess"
    ]
    
    def __init__(self, task_id: str, image_path: str, task_name: Optional[str] = None):
        self.task_id = task_id
        self.image_path = image_path
        self.task_name = task_name or f"task_{task_id[:8]}"
        self.created_at = datetime.now().isoformat()
        self.status = "created"  # created, processing, completed, failed
        self.current_step = None
        self.steps_completed = []
        self.error = None
        
        # Store intermediate results
        self.results = {}
        
        # Create output directory for this task
        self.output_dir = os.path.join(config.outputs_colored_dir, "debug", f"{task_id}_colored")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load and validate image
        try:
            self.original_image = Image.open(image_path)
            self.image_size = self.original_image.size
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "image_path": self.image_path,
            "image_size": self.image_size,
            "status": self.status,
            "current_step": self.current_step,
            "steps_completed": self.steps_completed,
            "created_at": self.created_at,
            "error": self.error,
            "output_dir": self.output_dir
        }
    
    def mark_step_complete(self, step: str, result: Any = None):
        """Mark a step as completed"""
        if step not in self.steps_completed:
            self.steps_completed.append(step)
        if result is not None:
            self.results[step] = result
        
        # Update current step to next
        try:
            current_idx = self.STEPS_ORDER.index(step)
            if current_idx + 1 < len(self.STEPS_ORDER):
                self.current_step = self.STEPS_ORDER[current_idx + 1]
            else:
                self.current_step = None
                self.status = "completed"
        except ValueError:
            pass
    
    def reset_to_step(self, step: str) -> List[str]:
        """Reset task to a specific step, clearing subsequent steps"""
        if step not in self.STEPS_ORDER:
            raise ValueError(f"Invalid step: {step}")
        
        target_idx = self.STEPS_ORDER.index(step)
        
        # Clear results for this step and all subsequent steps
        cleared_steps = []
        for i in range(target_idx, len(self.STEPS_ORDER)):
            step_name = self.STEPS_ORDER[i]
            if step_name in self.results:
                del self.results[step_name]
            if step_name in self.steps_completed:
                self.steps_completed.remove(step_name)
            cleared_steps.append(step_name)
        
        # Update current step
        self.current_step = step
        self.status = "processing"
        
        return cleared_steps
    
    def get_progress(self) -> float:
        """Calculate progress percentage"""
        return len(self.steps_completed) / len(self.STEPS_ORDER)


class TaskManager:
    """Manages all colorization tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
    
    def create_task(self, image_path: str, task_name: Optional[str] = None) -> Task:
        """Create a new task"""
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task
        task = Task(task_id, image_path, task_name)
        self.tasks[task_id] = task
        
        return task
    
    def get_task(self, task_id: str) -> Task:
        """Get task by ID"""
        if task_id not in self.tasks:
            raise KeyError(f"Task not found: {task_id}")
        return self.tasks[task_id]
    
    def list_tasks(self) -> List[Task]:
        """List all tasks"""
        return list(self.tasks.values())
    
    def delete_task(self, task_id: str):
        """Delete a task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
