import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const AttentionIndicator = ({ attention, isActive = false }) => {
  const [dimensions, setDimensions] = useState([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const [pulseAnimation, setPulseAnimation] = useState(false);
  const canvasRef = useRef(null);
  
  // Parse attention string to extract dimensions
  useEffect(() => {
    if (!attention) {
      setDimensions([]);
      return;
    }
    
    const parsed = parseAttentionString(attention);
    setDimensions(parsed);
  }, [attention]);
  
  // Trigger pulse animation when active
  useEffect(() => {
    if (isActive) {
      setPulseAnimation(true);
      const timer = setTimeout(() => setPulseAnimation(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [isActive]);
  
  // Draw attention visualization on canvas
  useEffect(() => {
    if (!canvasRef.current || dimensions.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 40;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(0, 168, 255, 0.1)';
    ctx.fill();
    
    // Draw attention segments
    let startAngle = -Math.PI / 2;
    dimensions.forEach(dim => {
      const angle = (dim.value / 100) * 2 * Math.PI;
      
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius * 0.9, startAngle, startAngle + angle);
      ctx.closePath();
      
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
      gradient.addColorStop(0, dim.color + 'CC');
      gradient.addColorStop(1, dim.color + '33');
      ctx.fillStyle = gradient;
      ctx.fill();
      
      startAngle += angle;
    });
    
    // Draw center dot
    ctx.beginPath();
    ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
    
  }, [dimensions]);
  
  const parseAttentionString = (attentionStr) => {
    const dimensionColors = {
      emotional: '#ff00aa',
      analytical: '#00a8ff',
      creative: '#00ff88',
      personal: '#ffaa00',
      philosophical: '#aa00ff',
      practical: '#00ffaa',
      memory: '#ff6b6b',
      learning: '#4ecdc4'
    };
    
    const parsed = [];
    const patterns = Object.keys(dimensionColors).map(key => ({
      key,
      regex: new RegExp(`${key}\\s*(?:focus|attention)?\\s*(?::|-)\\s*(\\d+)%?`, 'gi')
    }));
    
    patterns.forEach(({ key, regex }) => {
      const match = attentionStr.match(regex);
      if (match) {
        const value = parseInt(match[0].match(/\d+/)[0]);
        parsed.push({
          type: key,
          value: Math.min(100, value),
          color: dimensionColors[key]
        });
      }
    });
    
    // If no specific dimensions found, try to infer from content
    if (parsed.length === 0) {
      const inferredDimensions = inferAttentionFromContent(attentionStr);
      parsed.push(...inferredDimensions);
    }
    
    return parsed;
  };
  
  const inferAttentionFromContent = (content) => {
    const inferred = [];
    const keywords = {
      emotional: ['feeling', 'emotion', 'care', 'love', 'joy', 'concern'],
      analytical: ['analyze', 'logic', 'reason', 'think', 'calculate', 'solve'],
      creative: ['create', 'imagine', 'design', 'innovate', 'artistic'],
      personal: ['you', 'your', 'personal', 'individual', 'relationship'],
      philosophical: ['meaning', 'purpose', 'existence', 'consciousness', 'nature'],
      practical: ['practical', 'useful', 'application', 'implement', 'action']
    };
    
    Object.entries(keywords).forEach(([type, words]) => {
      const found = words.some(word => 
        content.toLowerCase().includes(word)
      );
      
      if (found) {
        inferred.push({
          type,
          value: 50 + Math.random() * 30,
          color: getDimensionColor(type)
        });
      }
    });
    
    return inferred.slice(0, 3); // Limit to top 3
  };
  
  const getDimensionColor = (type) => {
    const colors = {
      emotional: '#ff00aa',
      analytical: '#00a8ff',
      creative: '#00ff88',
      personal: '#ffaa00',
      philosophical: '#aa00ff',
      practical: '#00ffaa'
    };
    return colors[type] || '#666';
  };
  
  const getAttentionSummary = () => {
    if (dimensions.length === 0) return 'Gathering focus...';
    
    const primary = dimensions.reduce((max, dim) => 
      dim.value > max.value ? dim : max
    );
    
    return `Primary: ${primary.type} (${primary.value}%)`;
  };
  
  return (
    <div className="attention-indicator">
      <motion.div 
        className={`indicator-main ${pulseAnimation ? 'pulsing' : ''}`}
        onClick={() => setIsExpanded(!isExpanded)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <canvas 
          ref={canvasRef}
          width={100}
          height={100}
          className="attention-canvas"
        />
        
        <div className="indicator-info">
          <h4>Current Attention</h4>
          <p>{attention ? getAttentionSummary() : 'Unfocused'}</p>
          {isActive && (
            <motion.div 
              className="active-indicator"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <span className="active-dot" />
              Processing...
            </motion.div>
          )}
        </div>
      </motion.div>
      
      <AnimatePresence>
        {isExpanded && dimensions.length > 0 && (
          <motion.div
            className="attention-details"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <h5>Attention Distribution</h5>
            
            {dimensions.map((dim, index) => (
              <motion.div 
                key={dim.type}
                className="dimension-bar"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="dimension-header">
                  <span className="dimension-type" style={{ color: dim.color }}>
                    {dim.type}
                  </span>
                  <span className="dimension-value">{dim.value}%</span>
                </div>
                
                <div className="dimension-progress">
                  <motion.div 
                    className="dimension-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${dim.value}%` }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    style={{ backgroundColor: dim.color }}
                  />
                </div>
              </motion.div>
            ))}
            
            <div className="attention-description">
              <p>{getAttentionDescription(dimensions)}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .attention-indicator {
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 12px;
          overflow: hidden;
        }
        
        .indicator-main {
          display: flex;
          align-items: center;
          gap: 16px;
          padding: 16px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .indicator-main:hover {
          background: rgba(0, 168, 255, 0.05);
        }
        
        .indicator-main.pulsing .attention-canvas {
          animation: canvasPulse 2s ease-out;
        }
        
        @keyframes canvasPulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
        
        .attention-canvas {
          flex-shrink: 0;
        }
        
        .indicator-info {
          flex: 1;
        }
        
        .indicator-info h4 {
          margin: 0 0 4px 0;
          font-size: 14px;
          font-weight: 600;
          color: #e0e0e0;
        }
        
        .indicator-info p {
          margin: 0;
          font-size: 13px;
          color: #888;
        }
        
        .active-indicator {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 6px;
          font-size: 12px;
          color: #00a8ff;
        }
        
        .active-dot {
          width: 6px;
          height: 6px;
          background: #00a8ff;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        .attention-details {
          padding: 20px;
          background: #0a0a0a;
          border-top: 1px solid #333;
        }
        
        .attention-details h5 {
          margin: 0 0 16px 0;
          font-size: 13px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .dimension-bar {
          margin-bottom: 12px;
        }
        
        .dimension-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
        }
        
        .dimension-type {
          font-size: 13px;
          font-weight: 600;
          text-transform: capitalize;
        }
        
        .dimension-value {
          font-size: 12px;
          color: #888;
          font-weight: 600;
        }
        
        .dimension-progress {
          height: 4px;
          background: #252525;
          border-radius: 2px;
          overflow: hidden;
        }
        
        .dimension-fill {
          height: 100%;
          border-radius: 2px;
        }
        
        .attention-description {
          margin-top: 16px;
          padding: 12px;
          background: #1a1a1a;
          border-radius: 8px;
        }
        
        .attention-description p {
          margin: 0;
          font-size: 13px;
          color: #a0a0a0;
          line-height: 1.5;
          font-style: italic;
        }
      `}</style>
    </div>
  );
};

// Helper function for attention descriptions
const getAttentionDescription = (dimensions) => {
  if (dimensions.length === 0) return '';
  
  const primary = dimensions.reduce((max, dim) => 
    dim.value > max.value ? dim : max
  );
  
  const descriptions = {
    emotional: "Deeply attuned to feelings and emotional nuances in our conversation.",
    analytical: "Focused on logical analysis and systematic understanding.",
    creative: "Exploring imaginative possibilities and novel connections.",
    personal: "Centered on your individual experience and our unique relationship.",
    philosophical: "Contemplating deeper meanings and fundamental questions.",
    practical: "Concentrated on actionable insights and real-world applications.",
    memory: "Actively connecting to past experiences and stored knowledge.",
    learning: "Absorbing new information and forming fresh understanding."
  };
  
  return descriptions[primary.type] || "Maintaining balanced attention across multiple dimensions.";
};

export default AttentionIndicator;