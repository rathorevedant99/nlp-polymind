from typing import List, Dict, Any
import json
from pathlib import Path
import pandas as pd

class Memory:
    def __init__(self, config, save_dir: str = "./memory"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.feedback_history = []
        self.instruction_data = []
        self.config = config
        
    def add_critic_feedback(self, original_inputs: List[str], expert_outputs: List[str], critic_feedback: str, expert_id: int):
        """Store feedback from the critic during the debate phase."""
        feedback_entry = {
            "original_inputs": original_inputs,
            "expert_outputs": expert_outputs,
            "critic_feedback": critic_feedback,
            "expert_id": expert_id,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        self.feedback_history.append(feedback_entry)
        self._save_feedback()
        
    def format_instruction_data(self):
        """Convert stored feedback into instruction tuning format."""
        self.instruction_data = []
        for entry in self.feedback_history:
            for inp, out, feedback in zip(entry['original_inputs'], entry['expert_outputs'], entry['critic_feedback']):
                instruction = f"""Based on the critic's feedback, improve the response.
Input: {inp}
Previous Expert Response: {out}
Critic's Feedback: {feedback}
Generate an improved response that addresses the critic's feedback."""
                
                self.instruction_data.append({
                    "instruction": instruction,
                    "input": inp,
                    "output": feedback
                })

    def provide_instruction_data(self):
        """Provide instruction data for training."""
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
                
    def clear_memory(self):
        """Clear all stored feedback and instruction data."""
        self.feedback_history = []
        self.instruction_data = []
        self._save_feedback()