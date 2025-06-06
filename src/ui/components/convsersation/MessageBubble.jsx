import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const MessageBubble = ({ message }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [detectedActions, setDetectedActions] = useState([]);
  
  useEffect(() => {
    // Detect actions in the message content
    const actions = detectActions(message.content);
    setDetectedActions(actions);
  }, [message.content]);
  
  const detectActions = (content) => {
    const actions = [];
    
    // Pattern matching for different action types
    const patterns = {
      memory: /\b(remember|storing|saved|recorded)\b/i,
      emotion: /\b(feeling|emotion|mood)\b/i,
      goal: /\b(goal|objective|aim|plan)\b/i,
      attention: /\b(focus|attention|concentrat)/i,
      learning: /\b(learn|understand|realize|discover)\b/i
    };
    
    Object.entries(patterns).forEach(([type, pattern]) => {
      if (pattern.test(content)) {
        actions.push(type);
      }
    });
    
    return actions;
  };
  
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  const getActionIcon = (action) => {
    const icons = {
      memory: '💎',
      emotion: '💭',
      goal: '🎯',
      attention: '👁️',
      learning: '📚'
    };
    return icons[action] || '✨';
  };
  
  const renderContent = () => {
    if (!message.content) return null;
    
    // Split content into paragraphs
    const paragraphs = message.content.split('\n\n');
    
    return paragraphs.map((paragraph, index) => {
      // Check if it's a code block
      if (paragraph.startsWith('```')) {
        const lines = paragraph.split('\n');
        const language = lines[0].replace('```', '').trim();
        const code = lines.slice(1, -1).join('\n');
        
        return (
          <pre key={index} className="code-block">
            <div className="code-header">
              <span className="code-language">{language || 'code'}</span>
              <button 
                onClick={() => navigator.clipboard.writeText(code)}
                className="copy-button"
              >
                Copy
              </button>
            </div>
            <code>{code}</code>
          </pre>
        );
      }
      
      // Check if it's a list
      if (paragraph.includes('\n- ') || paragraph.includes('\n* ')) {
        const items = paragraph.split('\n').filter(line => line.trim());
        return (
          <ul key={index} className="message-list">
            {items.map((item, i) => (
              <li key={i}>{item.replace(/^[*-]\s*/, '')}</li>
            ))}
          </ul>
        );
      }
      
      // Regular paragraph with inline formatting
      return (
        <p key={index} className="message-paragraph">
          {formatInlineText(paragraph)}
        </p>
      );
    });
  };
  
  const formatInlineText = (text) => {
    // Simple inline formatting
    const formatted = text
      .split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/)
      .map((part, index) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={index}>{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith('*') && part.endsWith('*')) {
          return <em key={index}>{part.slice(1, -1)}</em>;
        }
        if (part.startsWith('`') && part.endsWith('`')) {
          return <code key={index} className="inline-code">{part.slice(1, -1)}</code>;
        }
        return part;
      });
    
    return formatted;
  };
  
  const isUser = message.role === 'user';
  const isError = message.role === 'error';
  const isThinking = message.role === 'thinking';
  
  return (
    <motion.div
      className={`message-bubble ${message.role}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="message-header">
        <span className="message-role">
          {isUser ? 'You' : isError ? 'System' : isThinking ? 'Thinking' : 'Claude'}
        </span>
        <span className="message-time">{formatTimestamp(message.timestamp)}</span>
      </div>
      
      <div className="message-content">
        {renderContent()}
      </div>
      
      {detectedActions.length > 0 && !isUser && (
        <div className="message-actions">
          <span className="actions-label">Actions detected:</span>
          <div className="action-tags">
            {detectedActions.map(action => (
              <span key={action} className={`action-tag ${action}`}>
                <span className="action-icon">{getActionIcon(action)}</span>
                {action}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {message.thinking && (
        <motion.div
          className="thinking-content"
          initial={{ height: 0 }}
          animate={{ height: isExpanded ? 'auto' : 0 }}
        >
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="thinking-toggle"
          >
            {isExpanded ? 'Hide' : 'Show'} thinking process
          </button>
          {isExpanded && (
            <div className="thinking-text">
              {message.thinking}
            </div>
          )}
        </motion.div>
      )}
      
      {message.attention_focus && !isUser && (
        <div className="attention-display">
          <span className="attention-label">Attention:</span>
          <span className="attention-focus">{message.attention_focus}</span>
        </div>
      )}
      
      <style jsx>{`
        .message-bubble {
          margin-bottom: 16px;
          animation: fadeInUp 0.3s ease-out;
        }
        
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .message-bubble.user {
          margin-left: auto;
          max-width: 70%;
        }
        
        .message-bubble.assistant {
          margin-right: auto;
          max-width: 80%;
        }
        
        .message-bubble.error {
          margin-right: auto;
          max-width: 80%;
        }
        
        .message-bubble.thinking {
          margin-right: auto;
          max-width: 80%;
          opacity: 0.7;
        }
        
        .message-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
          padding: 0 4px;
        }
        
        .message-role {
          font-size: 12px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .message-bubble.user .message-role {
          color: #00a8ff;
        }
        
        .message-bubble.error .message-role {
          color: #ff0066;
        }
        
        .message-time {
          font-size: 11px;
          color: #666;
        }
        
        .message-content {
          padding: 16px;
          background: #1a1a1a;
          border-radius: 12px;
          border: 1px solid #333;
        }
        
        .message-bubble.user .message-content {
          background: linear-gradient(135deg, #0a4d8a 0%, #0a3d6a 100%);
          border-color: #0a5d9a;
        }
        
        .message-bubble.error .message-content {
          background: rgba(255, 0, 102, 0.1);
          border-color: rgba(255, 0, 102, 0.3);
        }
        
        .message-paragraph {
          margin: 0 0 12px 0;
          line-height: 1.6;
          color: #e0e0e0;
        }
        
        .message-paragraph:last-child {
          margin-bottom: 0;
        }
        
        .message-list {
          margin: 0 0 12px 0;
          padding-left: 20px;
        }
        
        .message-list li {
          margin-bottom: 6px;
          color: #e0e0e0;
          line-height: 1.5;
        }
        
        .code-block {
          margin: 0 0 12px 0;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 8px;
          overflow: hidden;
        }
        
        .code-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: #252525;
          border-bottom: 1px solid #333;
        }
        
        .code-language {
          font-size: 12px;
          color: #00a8ff;
          font-weight: 600;
        }
        
        .copy-button {
          padding: 4px 8px;
          background: #333;
          border: none;
          border-radius: 4px;
          color: #888;
          font-size: 11px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .copy-button:hover {
          background: #444;
          color: #e0e0e0;
        }
        
        .code-block code {
          display: block;
          padding: 12px;
          font-family: 'SF Mono', Consolas, monospace;
          font-size: 13px;
          line-height: 1.5;
          color: #e0e0e0;
          overflow-x: auto;
        }
        
        .inline-code {
          padding: 2px 6px;
          background: rgba(0, 168, 255, 0.1);
          border: 1px solid rgba(0, 168, 255, 0.2);
          border-radius: 4px;
          font-family: 'SF Mono', Consolas, monospace;
          font-size: 0.9em;
          color: #00a8ff;
        }
        
        .message-actions {
          margin-top: 12px;
          padding: 12px;
          background: rgba(0, 168, 255, 0.05);
          border-radius: 8px;
          border: 1px solid rgba(0, 168, 255, 0.1);
        }
        
        .actions-label {
          display: block;
          font-size: 11px;
          color: #888;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 8px;
        }
        
        .action-tags {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }
        
        .action-tag {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 10px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 12px;
          font-size: 12px;
          color: #a0a0a0;
        }
        
        .action-tag.memory {
          border-color: #00ff88;
          color: #00ff88;
        }
        
        .action-tag.emotion {
          border-color: #ff00aa;
          color: #ff00aa;
        }
        
        .action-tag.goal {
          border-color: #ffaa00;
          color: #ffaa00;
        }
        
        .action-tag.attention {
          border-color: #00a8ff;
          color: #00a8ff;
        }
        
        .action-tag.learning {
          border-color: #aa00ff;
          color: #aa00ff;
        }
        
        .action-icon {
          font-size: 14px;
        }
        
        .thinking-content {
          margin-top: 12px;
          overflow: hidden;
        }
        
        .thinking-toggle {
          padding: 6px 12px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 6px;
          color: #888;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .thinking-toggle:hover {
          background: #333;
          color: #e0e0e0;
        }
        
        .thinking-text {
          margin-top: 8px;
          padding: 12px;
          background: rgba(108, 92, 231, 0.05);
          border: 1px solid rgba(108, 92, 231, 0.1);
          border-radius: 8px;
          font-size: 13px;
          color: #a0a0a0;
          line-height: 1.5;
          font-style: italic;
        }
        
        .attention-display {
          margin-top: 8px;
          padding: 8px 12px;
          background: rgba(0, 168, 255, 0.05);
          border-radius: 6px;
          font-size: 12px;
        }
        
        .attention-label {
          color: #888;
          font-weight: 600;
          margin-right: 8px;
        }
        
        .attention-focus {
          color: #00a8ff;
          font-style: italic;
        }
      `}</style>
    </motion.div>
  );
};

export default MessageBubble;