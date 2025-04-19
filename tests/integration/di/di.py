# tests/test_di_integration.py

def test_data_handler_integration():
    # Bootstrap with test configuration
    container, config = bootstrap(['config/test.yaml'])
    
    # Get components
    data_handler = container.get('data_handler')
    
    # Test functionality
    data_handler.load_data(['TEST'], '2022-01-01', '2022-12-31')
    bar = data_handler.get_next_bar('TEST')
    
    assert bar is not None
    # Other assertions
