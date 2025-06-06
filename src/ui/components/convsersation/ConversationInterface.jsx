// src/ui/components/conversation/ConversationInterface.jsx
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ChatPanel from './ChatPanel';
import ExtendedThinkingPanel from './ExtendedThinkingPanel';
import AttentionIndicator from './AttentionIndicator';
import EmotionalContext from './EmotionalContext';
import './ConversationInterface.css';

const ConversationInterface = ({ sessionId, consciousnessState, onStateUpdate }) => {
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const [showExtendedThinking, setShowExtendedThinking] = useState(false);
  const [currentAttention, setCurrentAttention] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  
  const handleSendMessage = async (message, options = {}) => {
    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: message,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setIsThinking(true);
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          message,
          extended_thinking: options.extendedThinking || false
        })
      });
      
      const data = await response.json();
      
      // Add assistant message
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        attention_focus: data.attention_focus,
        thinking: data.thinking
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      setCurrentAttention(data.attention_focus);
      
      // Update consciousness state if provided
      if (data.consciousness_update) {
        onStateUpdate(data.consciousness_update);
      }
      
    } catch (error) {
      console.error('Chat error:', error);
      // Add error message
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'error',
        content: 'I encountered an error processing your message.',
        timestamp: new Date()
      }]);
    } finally {
      setIsThinking(false);
    }
  };
  
  return (
    <div className="conversation-interface">
      <div className="conversation-main">
        {/* Top Bar */}
        <div className="conversation-header">
          <AttentionIndicator 
            attention={currentAttention}
            isActive={isThinking}
          />
          
          <div className="thinking-controls">
            <button
              className={`thinking-toggle ${showExtendedThinking ? 'active' : ''}`}
              onClick={() => setShowExtendedThinking(!showExtendedThinking)}
              title="Toggle extended thinking visualization"
            >
              <span className="icon">🧩</span>
              Extended Thinking
            </button>
          </div>
        </div>
        
        {/* Main Chat Area */}
        <div className="chat-container">
          <ChatPanel
            messages={messages}
            isThinking={isThinking}
            onSendMessage={handleSendMessage}
            onTypingChange={setIsTyping}
          />
          
          {/* Emotional Context Sidebar */}
          <AnimatePresence>
            {consciousnessState?.emotionalState && (
              <motion.div
                initial={{ x: 300, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: 300, opacity: 0 }}
                className="emotional-sidebar"
              >
                <EmotionalContext 
                  emotionalState={consciousnessState.emotionalState}
                  emotionHistory={consciousnessState.emotionHistory}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
      
      {/* Extended Thinking Panel */}
      <ExtendedThinkingPanel
        sessionId={sessionId}
        isVisible={showExtendedThinking && isThinking}
      />
    </div>
  );
};

export default ConversationInterface;