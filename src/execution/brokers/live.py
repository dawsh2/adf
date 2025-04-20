from typing import Dict, Any, Optional
import datetime
import logging

from src.core.events.event_types import OrderEvent, FillEvent
from src.core.events.event_utils import create_fill_event

class LiveBroker(BrokerBase):
    """
    Live broker implementation that connects to a real exchange/broker API.
    Translates internal orders to broker-specific formats.
    """
    
    def __init__(self, broker_api_client, fill_emitter=None, config=None):
        super().__init__(fill_emitter)
        self.broker_api = broker_api_client
        self.config = config or {}
        self.order_map = {}  # internal_id -> broker_id
        self.logger = logging.getLogger(__name__)
    
    def place_order(self, order):
        """Place an order with the live broker."""
        if not isinstance(order, OrderEvent):
            return
        
        try:
            # Extract order details
            symbol = order.get_symbol()
            order_type = order.get_order_type()
            direction = order.get_direction()
            quantity = order.get_quantity()
            price = order.get_price()
            
            # Translate to broker-specific format
            broker_order = self._translate_order(order)
            
            # Submit to broker API
            response = self.broker_api.submit_order(broker_order)
            
            # Store order mapping
            if response.get('success'):
                broker_order_id = response.get('order_id')
                self.order_map[order.get_id()] = broker_order_id
                
                self.logger.info(f"Order placed: {symbol} {direction} {quantity} @ {price}")
            else:
                self.logger.error(f"Order failed: {response.get('error')}")
        
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
    
    def cancel_order(self, order_id):
        """Cancel an existing order."""
        # Get broker's order id
        broker_order_id = self.order_map.get(order_id)
        if not broker_order_id:
            return False
        
        try:
            # Submit cancellation to broker API
            response = self.broker_api.cancel_order(broker_order_id)
            
            # Check response
            if response.get('success'):
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Cancel failed: {response.get('error')}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def _translate_order(self, order):
        """Translate internal order to broker-specific format."""
        # This would be customized for each broker API
        return {
            'symbol': order.get_symbol(),
            'quantity': order.get_quantity(),
            'side': order.get_direction().lower(),
            'type': order.get_order_type().lower(),
            'limit_price': order.get_price() if order.get_order_type() == 'LIMIT' else None,
            'time_in_force': 'day',
            'client_order_id': order.get_id()
        }
    
    def on_order_update(self, update):
        """Process order updates from the broker."""
        # Extract details from broker update
        broker_order_id = update.get('order_id')
        status = update.get('status')
        filled_qty = update.get('filled_quantity', 0)
        fill_price = update.get('fill_price')
        
        # Find internal order id
        internal_order_id = next(
            (int_id for int_id, brok_id in self.order_map.items() if brok_id == broker_order_id),
            None
        )
        
        if not internal_order_id:
            self.logger.warning(f"Received update for unknown order: {broker_order_id}")
            return
        
        # If this is a fill update
        if status in ['filled', 'partially_filled'] and filled_qty > 0 and fill_price:
            # Get the original order
            original_order = self.orders.get(internal_order_id)
            if not original_order:
                return
            
            # Create fill event
            fill = create_fill_event(
                symbol=original_order.get_symbol(),
                direction=original_order.get_direction(),
                quantity=filled_qty,
                price=fill_price,
                commission=update.get('commission', 0.0),
                timestamp=datetime.datetime.now()
            )
            
            # Emit fill event
            self.emit_fill(fill)
            
            self.logger.info(f"Fill received: {filled_qty} @ {fill_price}")
