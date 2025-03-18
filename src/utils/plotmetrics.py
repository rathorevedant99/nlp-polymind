import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.num_debate_rounds = len(metrics)

    def __call__(self):
        self.plot_all_metrics()
    
    def plot_all_metrics(self):
        self.plot_rouge_scores()
        self.plot_bertscore_scores()
        self.plot_novelty_scores()
        self.plot_length_ratios()
        
    def plot_rouge_scores(self):
        rouge_scores = [self.metrics[f"round_{i+1}"]["rouge_scores"] for i in range(self.num_debate_rounds)]
        plt.plot(rouge_scores)
        plt.show()

    def plot_bertscore_scores(self):
        bertscore_scores = [self.metrics[f"round_{i+1}"]["bertscore_scores"] for i in range(self.num_debate_rounds)]
        plt.plot(bertscore_scores)
        plt.show()

    def plot_novelty_scores(self):
        novelty_scores = [self.metrics[f"round_{i+1}"]["novelty_scores"] for i in range(self.num_debate_rounds)]
        plt.plot(novelty_scores)
        plt.show()

    def plot_length_ratios(self):
        length_ratios = [self.metrics[f"round_{i+1}"]["length_ratios"] for i in range(self.num_debate_rounds)]
        plt.plot(length_ratios)
        plt.show()
        
        
    
    