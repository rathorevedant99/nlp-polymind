"""
Evaluation metrics for the debate.
"""
from src.agent.team import ExpertTeam
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from nltk.corpus import stopwords
import nltk

class Metrics:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

    def __call__(self, ground_truth: str, expert_answers: dict):
        return self.calc_all_metrics(ground_truth, expert_answers)

    def _filter_stopwords(self, text: str) -> list:
        """Helper method to filter stopwords from text."""
        words = text.lower().split()
        return [word for word in words if word not in self.stop_words]
    
    def calc_all_metrics(self, ground_truth: str, expert_answers: dict):
        rouge_scores = self.eval_rouge(ground_truth, expert_answers)
        bertscore_scores = self.eval_bertscore(ground_truth, expert_answers)
        novelty_scores = self.eval_novelty(ground_truth, expert_answers)
        length_ratios = self.eval_length_ratio(ground_truth, expert_answers)
        return rouge_scores, bertscore_scores, novelty_scores, length_ratios

    def eval_rouge(self, ground_truth: str, expert_answers: dict):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for expert_id, answer in expert_answers.items():
            scores = scorer.score(ground_truth, answer)
            print(f"Expert {expert_id} rouge scores: {scores}")
        return scores
    
    def eval_bertscore(self, ground_truth: str, expert_answers: dict):
        # BERTScore handles stopwords contextually, no need for explicit filtering
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        for expert_id, answer in expert_answers.items():
            scores = scorer.score([ground_truth], [answer])
            print(f"Expert {expert_id} bertscore scores: {scores}")
        return scores
    
    def eval_novelty(self, ground_truth: str, expert_answers: dict):
        """
        Calculate novelty score by comparing unique words in expert answers vs ground truth.
        Returns ratio of unique words in expert answers not present in ground truth.
        Excludes stopwords from the calculation.
        """
        # Filter stopwords from ground truth
        truth_words = set(self._filter_stopwords(ground_truth))
        
        novelty_scores = {}
        for expert_id, answer in expert_answers.items():
            # Filter stopwords from expert answer
            expert_words = set(self._filter_stopwords(answer))
            
            # Calculate unique words (not in ground truth)
            unique_words = expert_words - truth_words
            
            # Calculate novelty score
            if len(expert_words) > 0:
                novelty_score = len(unique_words) / len(expert_words)
            else:
                novelty_score = 0.0
                
            novelty_scores[expert_id] = novelty_score
        
        return novelty_scores

    def eval_length_ratio(self, ground_truth: str, expert_answers: dict):
        """
        Calculate the ratio of expert answers length to ground truth length.
        Values > 1 indicate longer expert answers, < 1 indicate shorter answers.
        Excludes stopwords from the calculation.
        """
        truth_length = len(self._filter_stopwords(ground_truth))
        ratios = {}
        
        for expert_id, answer in expert_answers.items():
            expert_length = len(self._filter_stopwords(answer))
            ratios[expert_id] = expert_length / truth_length if truth_length > 0 else 0.0
        
        return ratios
    
