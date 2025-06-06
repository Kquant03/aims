import { useState, useEffect } from 'react';

const useWebSocket = (sessionId) => {
  const [messages, setMessages] = useState([]);
  const [connectionState, setConnectionState] = useState('connecting');
  
  const sendMessage = (message) => {
    console.log('Sending:', message);
  };
  
  return { messages, sendMessage, connectionState };
};

export default useWebSocket;
