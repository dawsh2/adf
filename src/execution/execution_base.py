class ExecutionEngineBase(ABC):
    @abstractmethod
    def on_signal(self, event):
        pass
    
    @abstractmethod
    def process_order(self, order):
        pass
