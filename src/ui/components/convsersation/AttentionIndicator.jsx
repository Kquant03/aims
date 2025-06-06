import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './AttentionIndicator.css';

const AttentionIndicator = ({ attention, isActive = false }) => {
  const [expanded, setExpanded] = useState(false);
  const [pulseIntensity, setPulseIntensity] = useState(0);
  const [attentionHistory, setAttentionHistory] = useState([]);
  const canvasRef = useRef(null);
  
  // Parse attention data
  const parseAttention = (attentionData) => {
    if (!attentionData) return null;
    
    if (typeof attentionData === 'string') {
      return {
        primary_focus: attentionData,
        dimensions: {},
        focus_type: 'general',
        confidence: 0.5
      };
    }
    
    return attentionData;
  };
  
  const currentAttention = parseAttention(attention);
  
  // Update attention history
  useEffect(() => {
    if (currentAttention && currentAttention.primary_focus) {
      setAttentionHistory(prev => [...prev.slice(-9), currentAttention]);
    }
  }, [currentAttention?.primary_focus]);
  
  // Calculate pulse intensity based on confidence
  useEffect(() => {
    if (currentAttention?.confidence) {
      setPulseIntensity(currentAttention.confidence);
    }
  }, [currentAttention?.confidence]);
  
  // Draw attention visualization
  useEffect(() => {
    if (!canvasRef.current || !currentAttention?.dimensions) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) / 2 - 10;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw dimension rays
    const dimensions = Object.entries(currentAttention.dimensions);
    const angleStep = (Math.PI * 2) / Math.max(dimensions.length, 1);
    
    dimensions.forEach(([key, value], index) => {
      const angle = angleStep * index - Math.PI / 2;
      const radius = maxRadius * value;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      // Draw ray
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = getColorForDimension(key);
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw point
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = getColorForDimension(key);
      ctx.fill();
      
      // Draw label
      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#888';
      ctx.textAlign = 'center';
      ctx.fillText(key, x, y - 8);
    });
    
    // Draw center point
    ctx.beginPath();
    ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#00ff88';
    ctx.fill();
    
    // Connect points to form shape
    if (dimensions.length > 2) {
      ctx.beginPath();
      dimensions.forEach(([key, value], index) => {
        const angle = angleStep * index - Math.PI / 2;
        const radius = maxRadius * value;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 255, 136, 0.1)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [currentAttention?.dimensions]);
  
  const getColorForDimension = (dimension) => {
    const colors = {
      novelty: '#00a8ff',
      relevance: '#00ff88',
      emotional: '#ff00aa',
      complexity: '#ffaa00',
      urgency: '#ff0066',
      personal: '#8a2be2'
    };
    return colors[dimension] || '#666';
  };
  
  const getFocusTypeIcon = (type) => {
    const icons = {
      emotional: '💭',
      analytical: '🔍',
      creative: '✨',
      memory: '💎',
      philosophical: '🌌',
      technical: '⚙️',
      personal: '❤️',
      general: '👁️'
    };
    return icons[type] || '👁️';
  };
  
  if (!currentAttention) {
    return (
      <div className="attention-indicator inactive">
        <div className="attention-icon">👁️</div>
        <span className="attention-text">No active focus</span>
      </div>
    );
  }
  
  return (
    <div className="attention-indicator-container">
      <motion.div
        className={`attention-indicator ${isActive ? 'active' : ''}`}
        onClick={() => setExpanded(!expanded)}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <motion.div
          className="attention-icon"
          animate={{
            scale: isActive ? [1, 1.2, 1] : 1,
            opacity: isActive ? [0.8, 1, 0.8] : 1
          }}
          transition={{
            duration: 2,
            repeat: isActive ? Infinity : 0,
            ease: "easeInOut"
          }}
        >
          {getFocusTypeIcon(currentAttention.focus_type)}
        </motion.div>
        
        <div className="attention-content">
          <div className="attention-focus">
            {currentAttention.primary_focus}
          </div>
          <div className="attention-meta">
            <span className="focus-type">{currentAttention.focus_type}</span>
            {currentAttention.confidence && (
              <span className="confidence">
                {(currentAttention.confidence * 100).toFixed(0)}% confident
              </span>
            )}
          </div>
        </div>
        
        <motion.div
          className="pulse-ring"
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.3, 0, 0.3]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeOut",
            times: [0, 0.5, 1]
          }}
          style={{
            opacity: pulseIntensity * 0.3
          }}
        />
      </motion.div>
      
      <AnimatePresence>
        {expanded && (
          <motion.div
            className="attention-details"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="dimensions-section">
              <h4>Attention Dimensions</h4>
              <canvas
                ref={canvasRef}
                width={200}
                height={150}
                className="dimensions-canvas"
              />
              
              <div className="dimension-list">
                {currentAttention.dimensions && Object.entries(currentAttention.dimensions).map(([key, value]) => (
                  <div key={key} className="dimension-item">
                    <span className="dimension-name">{key}</span>
                    <div className="dimension-bar">
                      <div
                        className="dimension-fill"
                        style={{
                          width: `${value * 100}%`,
                          backgroundColor: getColorForDimension(key)
                        }}
                      />
                    </div>
                    <span className="dimension-value">{(value * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
            
            {currentAttention.secondary_considerations && currentAttention.secondary_considerations.length > 0 && (
              <div className="secondary-section">
                <h4>Also Considering</h4>
                <div className="secondary-list">
                  {currentAttention.secondary_considerations.map((consideration, index) => (
                    <span key={index} className="secondary-item">
                      {consideration}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {attentionHistory.length > 1 && (
              <div className="history-section">
                <h4>Recent Focus</h4>
                <div className="history-list">
                  {attentionHistory.slice(-5).reverse().map((item, index) => (
                    <div key={index} className={`history-item ${index === 0 ? 'current' : ''}`}>
                      <span className="history-icon">
                        {getFocusTypeIcon(item.focus_type)}
                      </span>
                      <span className="history-text">
                        {item.primary_focus}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .attention-indicator-container {
          position: relative;
        }
        
        .attention-indicator {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 16px;
          background: rgba(26, 26, 26, 0.8);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          cursor: pointer;
          position: relative;
          overflow: hidden;
          transition: all 0.3s;
        }
        
        .attention-indicator:hover {
          background: rgba(26, 26, 26, 0.9);
          border-color: rgba(0, 255, 136, 0.3);
        }
        
        .attention-indicator.active {
          border-color: rgba(0, 255, 136, 0.5);
          box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        }
        
        .attention-indicator.inactive {
          opacity: 0.5;
          cursor: default;
        }
        
        .attention-icon {
          font-size: 24px;
          position: relative;
          z-index: 1;
        }
        
        .attention-content {
          flex: 1;
        }
        
        .attention-focus {
          color: #e0e0e0;
          font-size: 14px;
          line-height: 1.4;
        }
        
        .attention-meta {
          display: flex;
          gap: 12px;
          margin-top: 4px;
          font-size: 11px;
          color: #666;
        }
        
        .focus-type {
          text-transform: capitalize;
          color: #00ff88;
        }
        
        .confidence {
          color: #888;
        }
        
        .pulse-ring {
          position: absolute;
          top: 50%;
          left: 28px;
          width: 40px;
          height: 40px;
          border: 2px solid #00ff88;
          border-radius: 50%;
          transform: translate(-50%, -50%);
          pointer-events: none;
        }
        
        .attention-details {
          position: absolute;
          top: calc(100% + 8px);
          left: 0;
          right: 0;
          background: rgba(26, 26, 26, 0.95);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 16px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
          z-index: 100;
        }
        
        .dimensions-section h4,
        .secondary-section h4,
        .history-section h4 {
          margin: 0 0 12px 0;
          font-size: 12px;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .dimensions-canvas {
          width: 100%;
          height: 150px;
          margin-bottom: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 4px;
        }
        
        .dimension-list {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        
        .dimension-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 11px;
        }
        
        .dimension-name {
          width: 70px;
          color: #888;
          text-transform: capitalize;
        }
        
        .dimension-bar {
          flex: 1;
          height: 4px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
          overflow: hidden;
        }
        
        .dimension-fill {
          height: 100%;
          transition: width 0.3s ease;
        }
        
        .dimension-value {
          width: 35px;
          text-align: right;
          color: #e0e0e0;
        }
        
        .secondary-section {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .secondary-list {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }
        
        .secondary-item {
          padding: 4px 8px;
          background: rgba(0, 168, 255, 0.1);
          border: 1px solid rgba(0, 168, 255, 0.2);
          border-radius: 12px;
          font-size: 11px;
          color: #00a8ff;
        }
        
        .history-section {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .history-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        
        .history-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 4px 0;
          font-size: 12px;
          color: #666;
          transition: color 0.2s;
        }
        
        .history-item.current {
          color: #e0e0e0;
        }
        
        .history-icon {
          font-size: 14px;
          opacity: 0.6;
        }
        
        .history-text {
          flex: 1;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
      `}</style>
    </div>
  );
};

export default AttentionIndicator;