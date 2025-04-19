# main.py or equivalent

from core.bootstrap import bootstrap

def main():
    # Bootstrap the application
    container, config = bootstrap(['config/default.yaml', 'config/local.yaml'])
    
    # Get required components
    event_manager = container.get('event_manager')
    data_handler = container.get('data_handler')
    # Get other components as needed
    
    # Run your application
    # ...

if __name__ == "__main__":
    main()
