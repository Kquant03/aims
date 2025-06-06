import React, { useState } from 'react';
import { motion } from 'framer-motion';

const MemoryCard = ({ memory, onClick, isSelected = false, variant = 'default' }) => {
  const [isHovered, setIsHovered] = useState(false);
  
  const getImportanceColor = (importance) => {
    if (importance > 0.8) return '#ff0066';
    if (importance > 0.6) return '#ffaa00';
    if (importance > 0.4) return '#00a8ff';
    return '#666';
  };
  
  const getTypeIcon = (type) => {
    const icons = {
      conversation: '💬',
      insight: '💡',
      emotion: '💭',
      goal: '🎯',
      learning: '📚',
      milestone: '🏆',
      default: '💎'
    };
    return icons[type] || icons.default;
  };
  
  const getEmotionGradient = (emotion) => {
    const gradients = {
      joy: 'linear-gradient(135deg, #FFD93D 0%, #FFB347 100%)',
      curiosity: 'linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%)',
      calm: 'linear-gradient(135deg, #43E97B 0%, #38F9D7 100%)',
      excitement: 'linear-gradient(135deg, #FA709A 0%, #FEE140 100%)',
      concern: 'linear-gradient(135deg, #667EEA 0%, #764BA2 100%)',
      neutral: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    };
    return gradients[emotion] || gradients.neutral;
  };
  
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };
  
  const truncateContent = (content, maxLength = 150) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength).trim() + '...';
  };
  
  return (
    <motion.div
      className={`memory-card ${variant} ${isSelected ? 'selected' : ''}`}
      onClick={() => onClick(memory)}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Importance Indicator */}
      <div 
        className="importance-indicator"
        style={{ backgroundColor: getImportanceColor(memory.importance || 0.5) }}
      />
      
      {/* Header */}
      <div className="card-header">
        <div className="header-left">
          <span className="type-icon">{getTypeIcon(memory.type)}</span>
          <span className="memory-type">{memory.type || 'memory'}</span>
        </div>
        <span className="timestamp">{formatTimestamp(memory.timestamp)}</span>
      </div>
      
      {/* Content */}
      <div className="card-content">
        <p>{truncateContent(memory.content)}</p>
      </div>
      
      {/* Footer */}
      <div className="card-footer">
        {memory.emotional_context && (
          <div 
            className="emotion-badge"
            style={{ background: getEmotionGradient(memory.emotional_context.label) }}
          >
            <span>{memory.emotional_context.label}</span>
          </div>
        )}
        
        <div className="card-stats">
          {memory.associations && memory.associations.length > 0 && (
            <div className="stat-item">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="4" cy="4" r="2" stroke="currentColor" strokeWidth="1.5"/>
                <circle cx="10" cy="10" r="2" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M5.5 5.5L8.5 8.5" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
              <span>{memory.associations.length}</span>
            </div>
          )}
          
          <div className="stat-item">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M7 2L8.5 5.5L12 6L9.5 8.5L10 12L7 10L4 12L4.5 8.5L2 6L5.5 5.5L7 2Z" 
                    stroke="currentColor" strokeWidth="1.5" fill={memory.importance > 0.7 ? 'currentColor' : 'none'}/>
            </svg>
            <span>{(memory.importance * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
      
      {/* Hover Effect */}
      <motion.div 
        className="hover-effect"
        initial={{ opacity: 0 }}
        animate={{ opacity: isHovered ? 1 : 0 }}
        transition={{ duration: 0.2 }}
      />
      
      <style jsx>{`
        .memory-card {
          position: relative;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 12px;
          padding: 16px;
          cursor: pointer;
          transition: all 0.2s;
          overflow: hidden;
        }
        
        .memory-card:hover {
          border-color: #444;
          transform: translateY(-2px);
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        
        .memory-card.selected {
          border-color: #00a8ff;
          box-shadow: 0 0 0 2px rgba(0, 168, 255, 0.2);
        }
        
        .memory-card.compact {
          padding: 12px;
        }
        
        .memory-card.compact .card-content {
          font-size: 13px;
        }
        
        .importance-indicator {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          transition: height 0.2s;
        }
        
        .memory-card:hover .importance-indicator {
          height: 4px;
        }
        
        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        
        .header-left {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .type-icon {
          font-size: 16px;
        }
        
        .memory-type {
          font-size: 12px;
          font-weight: 600;
          text-transform: capitalize;
          color: #888;
        }
        
        .timestamp {
          font-size: 12px;
          color: #666;
        }
        
        .card-content {
          margin-bottom: 12px;
          color: #e0e0e0;
          font-size: 14px;
          line-height: 1.6;
        }
        
        .card-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
        }
        
        .emotion-badge {
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 600;
          color: white;
          text-transform: capitalize;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        
        .card-stats {
          display: flex;
          gap: 12px;
          margin-left: auto;
        }
        
        .stat-item {
          display: flex;
          align-items: center;
          gap: 4px;
          color: #666;
          font-size: 12px;
        }
        
        .stat-item svg {
          color: #666;
        }
        
        .hover-effect {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(135deg, transparent 0%, rgba(0, 168, 255, 0.05) 100%);
          pointer-events: none;
        }
        
        /* Variant styles */
        .memory-card.highlight {
          background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
          border-color: #444;
        }
        
        .memory-card.minimal {
          background: transparent;
          border: 1px solid #252525;
          padding: 12px;
        }
        
        .memory-card.minimal .card-content {
          margin-bottom: 8px;
        }
        
        .memory-card.minimal .emotion-badge {
          padding: 2px 8px;
          font-size: 10px;
        }
        
        /* Animation for new memories */
        @keyframes shimmer {
          0% {
            background-position: -200% center;
          }
          100% {
            background-position: 200% center;
          }
        }
        
        .memory-card.new::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(
            90deg,
            transparent,
            rgba(0, 168, 255, 0.1),
            transparent
          );
          background-size: 200% 100%;
          animation: shimmer 2s infinite;
          pointer-events: none;
        }
      `}</style>
    </motion.div>
  );
};

export default MemoryCard;