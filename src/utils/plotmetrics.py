import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.num_debate_rounds = len(metrics)
        
        # Extract ROUGE f-measure scores
        self.rouge_scores = [self.metrics[f"{i+1}"]["rouge_scores"]["rouge1"].fmeasure 
                           for i in range(self.num_debate_rounds)]
        
        # Extract BERTScore F1 scores (third tensor)
        self.bertscore_scores = [float(self.metrics[f"{i+1}"]["bertscore_scores"][2]) 
                                for i in range(self.num_debate_rounds)]
        
        # Extract average novelty scores (average of keys 0 and 1)
        self.novelty_scores = [sum(self.metrics[f"{i+1}"]["novelty_scores"].values()) / 2 
                              for i in range(self.num_debate_rounds)]
        
        # Extract average length ratios (average of keys 0 and 1)
        self.length_ratios = [sum(self.metrics[f"{i+1}"]["length_ratios"].values()) / 2 
                             for i in range(self.num_debate_rounds)]

    def __call__(self):
        self.plot_all_metrics()
    
    def plot_all_metrics(self):
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
        plt.show()
        
    def plot_rouge_scores(self):
        rounds = list(range(1, self.num_debate_rounds + 1))
        plt.plot(rounds, self.rouge_scores, marker='o')
        plt.title('ROUGE-1 F1 Scores')
        plt.xlabel('Debate Round')
        plt.ylabel('Score')
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
        
        
    
    