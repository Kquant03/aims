// src/ui/components/hooks/useWebSocket.js
import { useState, useEffect, useRef, useCallback } from 'react';

const useWebSocket = (sessionId) => {
  const [messages, setMessages] = useState([]);
  const [connectionState, setConnectionState] = useState('disconnected');
  const ws = useRef(null);
  const reconnectTimeout = useRef(null);
  const reconnectAttempts = useRef(0);
  
  const connect = useCallback(() => {
    if (!sessionId || ws.current?.readyState === WebSocket.OPEN) return;
    
    try {
      ws.current = new WebSocket(`ws://localhost:8765?session_id=${sessionId}`);
      
      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setConnectionState('connected');
        reconnectAttempts.current = 0;
      };
      
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Handle different message types
        switch (data.type) {
          case 'consciousness_update':
            // Broadcast consciousness updates
            window.dispatchEvent(new CustomEvent('consciousness_update', { 
              detail: data.data 
            }));
            break;
            
          case 'thinking_update':
            // Broadcast thinking updates
            window.dispatchEvent(new CustomEvent('thinking_update', { 
              detail: data.data 
            }));
            break;
            
          case 'memory_stored':
            // Broadcast memory updates
            window.dispatchEvent(new CustomEvent('memory_update', { 
              detail: data.data 
            }));
            break;
            
          default:
            setMessages(prev => [...prev, data]);
        }
      };
      
      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setConnectionState('disconnected');
        
        // Attempt reconnection
        if (reconnectAttempts.current < 5) {
          reconnectAttempts.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
          reconnectTimeout.current = setTimeout(connect, delay);
        }
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState('error');
      };
      
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionState('error');
    }
  }, [sessionId]);
  
  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connect]);
  
  const sendMessage = useCallback((message) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket not connected');
    }
  }, []);
  
  return { messages, sendMessage, connectionState };
};

export default useWebSocket;