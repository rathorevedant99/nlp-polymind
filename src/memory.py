from typing import List, Dict, Any
import json
from pathlib import Path
import pandas as pd

class Memory:
    def __init__(self, save_dir: str = "./memory", task_name: str = ""):
        self.save_dir = Path(save_dir) / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_history = []
        self.instruction_data = None
        
    def add_critic_feedback(self, original_inputs: List[str], expert_outputs: List[str], critic_feedback: str, append: bool = True):
        """Store feedback from the critic during the debate phase."""
        if append:
            self.load_feedback()
        
        feedback_entry = {
            "original_inputs": original_inputs,
            "expert_outputs": expert_outputs,
            "critic_feedback": critic_feedback,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Add to memory
        self.feedback_history.append(feedback_entry)
        
        # Save to file
        self._save_feedback()
    
    def format_instruction_data(self, load_all: bool = False):
        """
        Convert stored feedback into instruction tuning format.
        
        Args:
            load_all: If True, load all feedback from disk before formatting
        """
        self.instruction_data = []
        
        # If load_all is True, get all feedback from disk
        feedback_to_process = self.get_all_feedback() if load_all else self.feedback_history
        
        for entry in feedback_to_process:
            for inp, out, feedback in zip(entry['original_inputs'], entry['expert_outputs'], entry['critic_feedback']):
                instruction = f"""Below is feedback 
Input: {inp}
Previous Expert Response: {out}
Critic's Feedback: {feedback}
Generate an improved response that addresses the critic's feedback."""
                
                self.instruction_data.append({
                    "instruction": instruction,
                    "input": inp,
                    "output": feedback
                })

    def provide_instruction_data(self, load_all: bool = False):
        """
        Provide instruction data for training.
        
        Args:
            load_all: If True, load all feedback from disk before formatting
        """
        if self.instruction_data is None:
            self.format_instruction_data(load_all=load_all)
        return self.instruction_data
    
    def _save_feedback(self):
        """Save feedback history to disk."""
        feedback_file = self.save_dir / "feedback_history.json"
        with open(feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
            
    def load_feedback(self):
        """Load feedback history from disk."""
        feedback_file = self.save_dir / "feedback_history.json"
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                self.feedback_history = json.load(f)
                
    def clear_memory(self, append: bool = False):
        """Clear all stored feedback and instruction data."""
        self.feedback_history = []
        self.instruction_data = None
        # self._save_feedback(append=append)