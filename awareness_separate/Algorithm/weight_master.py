# weight_master.py

class WeightMaster:
    """
    Tracks and updates emotional parameters.
    """

    def __init__(self):
        self.awareness = 0.5
        self.curiosity = 0.5
        self.happiness = 0.5
        self.homeostatic_happiness = 0.5

        # Less strict threshold for demonstration
        self.tune_threshold = 0.45

    def update_parameters(self, incoherence_score: float):
        """
        Adjust stats based on incoherence_score (0.0 = fully coherent, 1.0 = fully incoherent).
        """
        self.awareness = min(1.0, self.awareness + 0.01)
        self.curiosity = min(1.0, self.curiosity + 0.005)
        self.happiness = self.homeostatic_happiness * (1.0 - incoherence_score)

    def should_finetune(self) -> bool:
        return self.happiness < self.tune_threshold