// src/ui/components/conversation/ChatPanel.jsx
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MessageBubble from './MessageBubble';
import './ChatPanel.css';

const ChatPanel = ({ messages, isThinking, onSendMessage, onTypingChange }) => {
  const [inputValue, setInputValue] = useState('');
  const [extendedThinking, setExtendedThinking] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isThinking) {
      onSendMessage(inputValue, { extendedThinking });
      setInputValue('');
      setExtendedThinking(false);
    }
  };
  
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
    onTypingChange(e.target.value.length > 0);
  };
  
  return (
    <div className="chat-panel">
      <div className="messages-container">
        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </AnimatePresence>
        
        {isThinking && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="thinking-indicator"
          >
            <div className="thinking-dots">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
            <span className="thinking-text">Processing...</span>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="input-container">
          <div className="input-options">
            <button
              type="button"
              className={`option-button ${extendedThinking ? 'active' : ''}`}
              onClick={() => setExtendedThinking(!extendedThinking)}
              title="Enable extended thinking"
            >
              🧠
            </button>
          </div>
          
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Share your thoughts..."
            disabled={isThinking}
            className="chat-input"
          />
          
          <button 
            type="submit" 
            disabled={!inputValue.trim() || isThinking}
            className="send-button"
          >
            <span className="send-icon">→</span>
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatPanel;