import matplotlib.pyplot as plt
import os
class Plotter:
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.num_debate_rounds = len(metrics)
        
        self.rouge1_scores = [self.metrics[f"{i+1}"]["rouge_scores"]["rouge1"].fmeasure 
                           for i in range(self.num_debate_rounds)]
        self.rouge2_scores = [self.metrics[f"{i+1}"]["rouge_scores"]["rouge2"].fmeasure 
                           for i in range(self.num_debate_rounds)]
        self.rougeL_scores = [self.metrics[f"{i+1}"]["rouge_scores"]["rougeL"].fmeasure 
                           for i in range(self.num_debate_rounds)]
        
        self.bertscore_scores = [float(self.metrics[f"{i+1}"]["bertscore_scores"][2]) 
                                for i in range(self.num_debate_rounds)]
        
        self.novelty_scores = [sum(self.metrics[f"{i+1}"]["novelty_scores"].values()) / 2 
                              for i in range(self.num_debate_rounds)]
        
        self.length_ratios = [sum(self.metrics[f"{i+1}"]["length_ratios"].values()) / 2 
                             for i in range(self.num_debate_rounds)]

    def __call__(self, save_path: str = None):
        self.plot_all_metrics(save_path)
    
    def plot_all_metrics(self, save_path: str = None):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        self.plot_rouge_scores()
        
        plt.subplot(2, 2, 2)
        self.plot_bertscore_scores()
        
        plt.subplot(2, 2, 3)
        self.plot_novelty_scores()
        
        plt.subplot(2, 2, 4)
        self.plot_length_ratios()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "metrics.png"))
        plt.show()
        
    def plot_rouge_scores(self):
        rounds = list(range(1, self.num_debate_rounds + 1))
        plt.plot(rounds, self.rouge1_scores, marker='o', label='ROUGE-1 F1')
        plt.plot(rounds, self.rouge2_scores, marker='o', label='ROUGE-2 F1')
        plt.plot(rounds, self.rougeL_scores, marker='o', label='ROUGE-L F1')
        plt.title('ROUGE Scores')
        plt.xlabel('Debate Round')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

    def plot_bertscore_scores(self):
        rounds = list(range(1, self.num_debate_rounds + 1))
        plt.plot(rounds, self.bertscore_scores, marker='o')
        plt.title('BERTScore F1')
        plt.xlabel('Debate Round')
        plt.ylabel('Score')
        plt.grid(True)

    def plot_novelty_scores(self):
        rounds = list(range(1, self.num_debate_rounds + 1))
        plt.plot(rounds, self.novelty_scores, marker='o')
        plt.title('Average Novelty Scores')
        plt.xlabel('Debate Round')
        plt.ylabel('Score')
        plt.grid(True)

    def plot_length_ratios(self):
        rounds = list(range(1, self.num_debate_rounds + 1))
        plt.plot(rounds, self.length_ratios, marker='o')
        plt.title('Average Length Ratios')
        plt.xlabel('Debate Round')
        plt.ylabel('Ratio')
        plt.grid(True)
        
        
    
    