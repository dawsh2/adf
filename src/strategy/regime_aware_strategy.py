import numpy as np

class EnsembleStrategy:
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        self.strategy_names = [s.name for s in strategies]
        self.weights = weights or [1 / len(strategies)] * len(strategies)

    def generate_signals(self, data):
        combined_signals = sum(
            weight * strategy.generate_signals(data)
            for strategy, weight in zip(self.strategies, self.weights)
        )
        return combined_signals

    def get_parameters(self):
        # Return a flat param space using strategy names
        return {
            f'weight_{name}': list(np.linspace(0.0, 1.0, 6))
            for name in self.strategy_names
        }

    def set_parameters(self, params):
        # Rebuild weights from param dict using strategy names
        weights = [params[f'weight_{name}'] for name in self.strategy_names]
        total = sum(weights)
        self.weights = [w / total for w in weights] if total > 0 else weights

