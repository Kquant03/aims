import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import './MessageBubble.css';

const MessageBubble = ({ message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showActions, setShowActions] = useState(false);
  
  const isUser = message.role === 'user';
  const isError = message.role === 'error';
  const hasThinking = message.thinking && message.thinking.length > 0;
  const hasAttention = message.attention_focus && message.attention_focus.length > 0;
  
  // Parse message for action indicators
  const parseActions = (content) => {
    const actionPatterns = [
      { pattern: /I'll remember/gi, type: 'memory', icon: '💎' },
      { pattern: /This reminds me/gi, type: 'recall', icon: '🔍' },
      { pattern: /I'm feeling/gi, type: 'emotion', icon: '💭' },
      { pattern: /Let me think/gi, type: 'thinking', icon: '🧠' },
      { pattern: /I'll focus on/gi, type: 'focus', icon: '🎯' }
    ];
    
    const detectedActions = [];
    actionPatterns.forEach(({ pattern, type, icon }) => {
      if (pattern.test(content)) {
        detectedActions.push({ type, icon });
      }
    });
    
    return detectedActions;
  };
  
  const actions = parseActions(message.content);
  
  // Format timestamp
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Highlight action phrases in content
  const highlightContent = (content) => {
    let highlighted = content;
    
    // Highlight action phrases
    const highlights = [
      { pattern: /(I'll remember.*?)(?=\.|$)/gi, className: 'action-memory' },
      { pattern: /(This reminds me.*?)(?=\.|$)/gi, className: 'action-recall' },
      { pattern: /(I'm feeling.*?)(?=\.|$)/gi, className: 'action-emotion' },
      { pattern: /(Let me think.*?)(?=\.|$)/gi, className: 'action-thinking' },
      { pattern: /(I'll focus on.*?)(?=\.|$)/gi, className: 'action-focus' }
    ];
    
    highlights.forEach(({ pattern, className }) => {
      highlighted = highlighted.replace(pattern, `<span class="${className}">$1</span>`);
    });
    
    return { __html: highlighted };
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className={`message-bubble ${isUser ? 'user' : 'assistant'} ${isError ? 'error' : ''}`}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className="message-header">
        <span className="message-role">
          {isUser ? '👤' : '🤖'} {isUser ? 'You' : 'AIMS'}
        </span>
        <span className="message-time">{formatTime(message.timestamp)}</span>
      </div>
      
      <div className="message-content">
        {isError ? (
          <div className="error-content">{message.content}</div>
        ) : (
          <div 
            className="text-content"
            dangerouslySetInnerHTML={highlightContent(message.content)}
          />
        )}
      </div>
      
      {hasAttention && (
        <div className="attention-indicator">
          <span className="attention-icon">👁️</span>
          <span className="attention-text">{message.attention_focus}</span>
        </div>
      )}
      
      {actions.length > 0 && (
        <motion.div 
          className="action-indicators"
          initial={{ opacity: 0 }}
          animate={{ opacity: showActions ? 1 : 0.5 }}
        >
          {actions.map((action, index) => (
            <span key={index} className={`action-badge ${action.type}`}>
              <span className="action-icon">{action.icon}</span>
              <span className="action-type">{action.type}</span>
            </span>
          ))}
        </motion.div>
      )}
      
      {hasThinking && (
        <div className="thinking-section">
          <button
            className="thinking-toggle"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
            <span className="toggle-text">Extended Thinking</span>
            <span className="thinking-badge">🧠</span>
          </button>
          
          <motion.div
            initial={false}
            animate={{ height: isExpanded ? 'auto' : 0 }}
            transition={{ duration: 0.3 }}
            className="thinking-content"
          >
            <div className="thinking-inner">
              {message.thinking}
            </div>
          </motion.div>
        </div>
      )}
      
      <style jsx>{`
        .message-bubble {
          margin: 12px 0;
          padding: 16px;
          border-radius: 12px;
          position: relative;
          transition: all 0.2s ease;
        }
        
        .message-bubble.user {
          background: linear-gradient(135deg, #1a3a52 0%, #0a2940 100%);
          margin-left: 20%;
          border: 1px solid rgba(0, 168, 255, 0.2);
        }
        
        .message-bubble.assistant {
          background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
          margin-right: 20%;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .message-bubble.error {
          background: linear-gradient(135deg, #3a1a1a 0%, #2a0a0a 100%);
          border: 1px solid rgba(255, 0, 102, 0.3);
        }
        
        .message-bubble:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }
        
        .message-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
          font-size: 12px;
          color: #888;
        }
        
        .message-role {
          display: flex;
          align-items: center;
          gap: 4px;
          font-weight: 600;
        }
        
        .message-time {
          color: #666;
        }
        
        .message-content {
          color: #e0e0e0;
          line-height: 1.6;
          word-wrap: break-word;
        }
        
        .text-content {
          white-space: pre-wrap;
        }
        
        /* Action highlights */
        .action-memory {
          background: rgba(0, 255, 136, 0.1);
          color: #00ff88;
          padding: 2px 4px;
          border-radius: 4px;
          font-weight: 500;
        }
        
        .action-recall {
          background: rgba(0, 168, 255, 0.1);
          color: #00a8ff;
          padding: 2px 4px;
          border-radius: 4px;
          font-weight: 500;
        }
        
        .action-emotion {
          background: rgba(255, 0, 170, 0.1);
          color: #ff00aa;
          padding: 2px 4px;
          border-radius: 4px;
          font-weight: 500;
        }
        
        .action-thinking {
          background: rgba(255, 170, 0, 0.1);
          color: #ffaa00;
          padding: 2px 4px;
          border-radius: 4px;
          font-weight: 500;
        }
        
        .action-focus {
          background: rgba(138, 43, 226, 0.1);
          color: #8a2be2;
          padding: 2px 4px;
          border-radius: 4px;
          font-weight: 500;
        }
        
        .attention-indicator {
          margin-top: 12px;
          padding: 8px 12px;
          background: rgba(0, 168, 255, 0.1);
          border: 1px solid rgba(0, 168, 255, 0.2);
          border-radius: 8px;
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: #00a8ff;
        }
        
        .attention-icon {
          font-size: 16px;
        }
        
        .action-indicators {
          margin-top: 12px;
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }
        
        .action-badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 8px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          font-size: 11px;
          color: #888;
          transition: all 0.2s;
        }
        
        .action-badge:hover {
          background: rgba(255, 255, 255, 0.1);
          color: #e0e0e0;
        }
        
        .action-badge.memory {
          border-color: rgba(0, 255, 136, 0.3);
          color: #00ff88;
        }
        
        .action-badge.recall {
          border-color: rgba(0, 168, 255, 0.3);
          color: #00a8ff;
        }
        
        .action-badge.emotion {
          border-color: rgba(255, 0, 170, 0.3);
          color: #ff00aa;
        }
        
        .action-badge.thinking {
          border-color: rgba(255, 170, 0, 0.3);
          color: #ffaa00;
        }
        
        .action-badge.focus {
          border-color: rgba(138, 43, 226, 0.3);
          color: #8a2be2;
        }
        
        .action-icon {
          font-size: 14px;
        }
        
        .thinking-section {
          margin-top: 12px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          padding-top: 12px;
        }
        
        .thinking-toggle {
          display: flex;
          align-items: center;
          gap: 8px;
          background: none;
          border: none;
          color: #888;
          font-size: 12px;
          cursor: pointer;
          padding: 4px 8px;
          margin: -4px -8px;
          border-radius: 6px;
          transition: all 0.2s;
          width: 100%;
          text-align: left;
        }
        
        .thinking-toggle:hover {
          background: rgba(255, 255, 255, 0.05);
          color: #e0e0e0;
        }
        
        .toggle-icon {
          font-size: 10px;
          transition: transform 0.2s;
        }
        
        .thinking-badge {
          margin-left: auto;
          font-size: 16px;
        }
        
        .thinking-content {
          overflow: hidden;
        }
        
        .thinking-inner {
          padding: 12px;
          margin-top: 8px;
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          font-size: 13px;
          color: #a0a0a0;
          line-height: 1.6;
        }
        
        .error-content {
          color: #ff6666;
        }
      `}</style>
    </motion.div>
  );
};

export default MessageBubble;