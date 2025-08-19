#!/usr/bin/env python3
"""
README Todo Updater for Pradel-JAX

This script helps maintain the todo list in README.md by:
1. Tracking completion status
2. Updating dates and progress
3. Moving completed items to the "Recently Completed" section
4. Adding new priority items

Usage:
    python update_readme_todos.py --complete "task_name" --date "2025-08-14"
    python update_readme_todos.py --add "New task description" --priority "high"
    python update_readme_todos.py --update-status "Enhanced documentation and RMark validation"
"""

import re
from datetime import datetime
from pathlib import Path
import argparse


class ReadmeTodoUpdater:
    def __init__(self, readme_path="README.md"):
        self.readme_path = Path(readme_path)
        self.content = self.readme_path.read_text()
    
    def update_last_modified(self):
        """Update the 'Last Updated' date to today."""
        today = datetime.now().strftime("%B %d, %Y")
        pattern = r"(> \*\*📅 Last Updated:\*\*) [^\\n]+"
        replacement = f"\\1 {today}"
        self.content = re.sub(pattern, replacement, self.content)
    
    def update_current_focus(self, focus_text):
        """Update the current focus section."""
        pattern = r"(\*\*🚧 Current Focus:\*\*) [^\\n]+"
        replacement = f"\\1 {focus_text}"
        self.content = re.sub(pattern, replacement, self.content)
    
    def mark_task_complete(self, task_number, completion_date=None):
        """Mark a task as completed and move it to the completed section."""
        if completion_date is None:
            completion_date = datetime.now().strftime("%B %d, %Y")
        
        # Find the task to move
        pattern = rf"^{task_number}\. \*\*[^*]+\*\* - (.+)$"
        match = re.search(pattern, self.content, re.MULTILINE)
        
        if match:
            task_description = match.group(1)
            # Remove from todo list
            self.content = re.sub(f"^{task_number}\\. \\*\\*[^*]+\\*\\* - .+\\n", "", self.content, flags=re.MULTILINE)
            
            # Add to completed section
            completed_pattern = r"(\*\*✅ Recently Completed:\*\*\n)(- [^\n]+\n)*"
            new_completed = f"\\1- {task_description} *({completion_date})*\n\\2"
            self.content = re.sub(completed_pattern, new_completed, self.content)
            
            # Renumber remaining tasks
            self._renumber_tasks()
    
    def add_new_task(self, description, priority="medium"):
        """Add a new task to the appropriate priority section."""
        if priority.lower() == "high":
            section_pattern = r"(### ⭐ High Priority \(Next 2-3 weeks\)\n\n)((?:\d+\. [^\n]+\n)*)"
            task_number = self._get_next_task_number("high")
        elif priority.lower() == "medium":
            section_pattern = r"(### 🔧 Medium Priority \(Next 1-2 months\)\n\n)((?:\d+\. [^\n]+\n)*)"
            task_number = self._get_next_task_number("medium")
        else:  # low priority
            section_pattern = r"(### 🎯 Lower Priority \(Next 2-3 months\)\n\n)((?:\d+\. [^\n]+\n)*)"
            task_number = self._get_next_task_number("low")
        
        new_task = f"{task_number}. **📋 {description.split(' - ')[0]}** - {description}\n"
        replacement = f"\\1{new_task}\\2"
        self.content = re.sub(section_pattern, replacement, self.content)
        self._renumber_tasks()
    
    def _get_next_task_number(self, priority):
        """Get the next available task number for a priority section."""
        # Find the highest current number
        pattern = r"^(\d+)\."
        numbers = [int(match.group(1)) for match in re.finditer(pattern, self.content, re.MULTILINE)]
        return max(numbers) + 1 if numbers else 1
    
    def _renumber_tasks(self):
        """Renumber all tasks in sequential order."""
        def replace_number(match):
            nonlocal counter
            counter += 1
            return f"{counter}. "
        
        counter = 0
        pattern = r"^\d+\. "
        self.content = re.sub(pattern, replace_number, self.content, flags=re.MULTILINE)
    
    def save(self):
        """Save the updated README."""
        self.readme_path.write_text(self.content)
        print(f"✅ Updated {self.readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Update README todos")
    parser.add_argument("--complete", type=int, help="Mark task number as complete")
    parser.add_argument("--date", help="Completion date (default: today)")
    parser.add_argument("--add", help="Add new task description")
    parser.add_argument("--priority", choices=["high", "medium", "low"], default="medium", 
                       help="Priority for new task")
    parser.add_argument("--focus", help="Update current focus text")
    parser.add_argument("--update-date", action="store_true", help="Update last modified date")
    
    args = parser.parse_args()
    
    updater = ReadmeTodoUpdater()
    
    if args.complete:
        updater.mark_task_complete(args.complete, args.date)
        print(f"✅ Marked task {args.complete} as complete")
    
    if args.add:
        updater.add_new_task(args.add, args.priority)
        print(f"✅ Added new {args.priority} priority task")
    
    if args.focus:
        updater.update_current_focus(args.focus)
        print(f"✅ Updated current focus")
    
    if args.update_date or args.complete or args.add or args.focus:
        updater.update_last_modified()
        updater.save()
    
    if not any([args.complete, args.add, args.focus, args.update_date]):
        print("📋 Current README todos are up to date")
        print("Use --help to see available update options")


if __name__ == "__main__":
    main()