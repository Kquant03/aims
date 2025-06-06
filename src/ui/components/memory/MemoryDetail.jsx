import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const MemoryDetail = ({ memory, onClose }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [detailedData, setDetailedData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    if (memory && !detailedData) {
      fetchDetailedMemory();
    }
  }, [memory]);
  
  const fetchDetailedMemory = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/memories/${memory.id}`, {
        credentials: 'include'
      });
      const data = await response.json();
      setDetailedData(data);
    } catch (error) {
      console.error('Failed to fetch memory details:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  if (!memory) return null;
  
  const getEmotionColor = (emotion) => {
    const colors = {
      joy: '#ffeb3b',
      curiosity: '#03a9f4',
      calm: '#4caf50',
      excitement: '#ff5722',
      concern: '#9c27b0',
      neutral: '#607d8b'
    };
    return colors[emotion] || colors.neutral;
  };
  
  const getImportanceLevel = (importance) => {
    if (importance > 0.8) return { label: 'Critical', color: '#ff0066' };
    if (importance > 0.6) return { label: 'High', color: '#ffaa00' };
    if (importance > 0.4) return { label: 'Medium', color: '#00a8ff' };
    return { label: 'Low', color: '#666' };
  };
  
  const importanceInfo = getImportanceLevel(memory.importance || 0.5);
  
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="memory-detail-overlay"
        onClick={onClose}
      >
        <motion.div
          initial={{ x: 400, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 400, opacity: 0 }}
          transition={{ type: 'spring', damping: 20 }}
          className={`memory-detail-panel ${isExpanded ? 'expanded' : ''}`}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="detail-header">
            <div className="header-content">
              <h3>Memory Details</h3>
              <span className="memory-id">#{memory.id.slice(0, 8)}</span>
            </div>
            <div className="header-actions">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="expand-button"
                title={isExpanded ? 'Collapse' : 'Expand'}
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d={isExpanded ? "M6 2L2 2V6M14 2L14 6H10M2 14L2 10H6M10 14H14V10" : "M2 6V2H6M10 2H14V6M6 14H2V10M14 10V14H10"} stroke="currentColor" strokeWidth="2"/>
                </svg>
              </button>
              <button onClick={onClose} className="close-button">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M4 4L12 12M4 12L12 4" stroke="currentColor" strokeWidth="2"/>
                </svg>
              </button>
            </div>
          </div>
          
          {/* Content */}
          <div className="detail-content">
            {isLoading ? (
              <div className="loading-state">
                <div className="loader" />
                <p>Loading memory details...</p>
              </div>
            ) : (
              <>
                {/* Memory Content */}
                <div className="content-section">
                  <h4>Content</h4>
                  <div className="memory-content">
                    {memory.content}
                  </div>
                </div>
                
                {/* Metadata */}
                <div className="metadata-section">
                  <div className="metadata-grid">
                    <div className="metadata-item">
                      <label>Timestamp</label>
                      <span>{new Date(memory.timestamp).toLocaleString()}</span>
                    </div>
                    
                    <div className="metadata-item">
                      <label>Type</label>
                      <span className="memory-type">{memory.type || 'conversation'}</span>
                    </div>
                    
                    <div className="metadata-item">
                      <label>Importance</label>
                      <div className="importance-display">
                        <div 
                          className="importance-bar"
                          style={{ 
                            width: `${(memory.importance || 0.5) * 100}%`,
                            backgroundColor: importanceInfo.color
                          }}
                        />
                        <span className="importance-label" style={{ color: importanceInfo.color }}>
                          {importanceInfo.label} ({((memory.importance || 0.5) * 100).toFixed(0)}%)
                        </span>
                      </div>
                    </div>
                    
                    {detailedData?.current_salience !== undefined && (
                      <div className="metadata-item">
                        <label>Current Salience</label>
                        <span>{(detailedData.current_salience * 100).toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Emotional Context */}
                {(memory.emotional_context || detailedData?.emotional_context) && (
                  <div className="emotion-section">
                    <h4>Emotional Context</h4>
                    <div className="emotion-display">
                      <div 
                        className="emotion-indicator"
                        style={{ 
                          backgroundColor: getEmotionColor(
                            memory.emotional_context?.label || 
                            detailedData?.emotional_context?.label || 
                            'neutral'
                          ) 
                        }}
                      />
                      <div className="emotion-values">
                        <div className="emotion-metric">
                          <label>Pleasure</label>
                          <div className="metric-bar">
                            <div 
                              className="metric-fill"
                              style={{ 
                                width: `${((memory.emotional_context?.pleasure || detailedData?.emotional_context?.pleasure || 0.5) * 100)}%`,
                                backgroundColor: '#ff6b6b'
                              }}
                            />
                          </div>
                        </div>
                        <div className="emotion-metric">
                          <label>Arousal</label>
                          <div className="metric-bar">
                            <div 
                              className="metric-fill"
                              style={{ 
                                width: `${((memory.emotional_context?.arousal || detailedData?.emotional_context?.arousal || 0.5) * 100)}%`,
                                backgroundColor: '#4ecdc4'
                              }}
                            />
                          </div>
                        </div>
                        <div className="emotion-metric">
                          <label>Dominance</label>
                          <div className="metric-bar">
                            <div 
                              className="metric-fill"
                              style={{ 
                                width: `${((memory.emotional_context?.dominance || detailedData?.emotional_context?.dominance || 0.5) * 100)}%`,
                                backgroundColor: '#45b7d1'
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Associations */}
                {detailedData?.associations && detailedData.associations.length > 0 && (
                  <div className="associations-section">
                    <h4>Associated Memories</h4>
                    <div className="associations-list">
                      {detailedData.associations.map((assoc, index) => (
                        <div key={index} className="association-item">
                          <span className="association-strength">{(assoc.strength * 100).toFixed(0)}%</span>
                          <span className="association-content">{assoc.content.substring(0, 100)}...</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Decay Information */}
                {detailedData?.decay_rate !== undefined && (
                  <div className="decay-section">
                    <h4>Memory Persistence</h4>
                    <div className="decay-info">
                      <p>Decay Rate: {(detailedData.decay_rate * 100).toFixed(2)}% per day</p>
                      <p>Estimated retention: {Math.floor(100 / detailedData.decay_rate)} days</p>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </motion.div>
      </motion.div>
      
      <style jsx>{`
        .memory-detail-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          z-index: 1000;
          display: flex;
          justify-content: flex-end;
        }
        
        .memory-detail-panel {
          width: 480px;
          height: 100%;
          background: #0a0a0a;
          border-left: 1px solid #333;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          transition: width 0.3s ease;
        }
        
        .memory-detail-panel.expanded {
          width: 720px;
        }
        
        .detail-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          border-bottom: 1px solid #333;
        }
        
        .header-content {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .header-content h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }
        
        .memory-id {
          font-size: 12px;
          color: #666;
          font-family: 'SF Mono', monospace;
        }
        
        .header-actions {
          display: flex;
          gap: 8px;
        }
        
        .expand-button, .close-button {
          padding: 8px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 6px;
          color: #666;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .expand-button:hover, .close-button:hover {
          background: #252525;
          color: #e0e0e0;
          border-color: #444;
        }
        
        .detail-content {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
        }
        
        .loading-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 200px;
          color: #666;
        }
        
        .loader {
          width: 40px;
          height: 40px;
          border: 3px solid #333;
          border-top-color: #00a8ff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 16px;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .content-section, .metadata-section, .emotion-section, 
        .associations-section, .decay-section {
          margin-bottom: 24px;
          padding-bottom: 24px;
          border-bottom: 1px solid #1a1a1a;
        }
        
        .content-section:last-child, .metadata-section:last-child, 
        .emotion-section:last-child, .associations-section:last-child, 
        .decay-section:last-child {
          border-bottom: none;
        }
        
        h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .memory-content {
          background: #1a1a1a;
          padding: 16px;
          border-radius: 8px;
          line-height: 1.6;
          color: #e0e0e0;
        }
        
        .metadata-grid {
          display: grid;
          gap: 16px;
        }
        
        .metadata-item {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        
        .metadata-item label {
          font-size: 12px;
          color: #666;
          font-weight: 600;
        }
        
        .metadata-item span {
          color: #e0e0e0;
          font-size: 14px;
        }
        
        .memory-type {
          display: inline-block;
          padding: 4px 8px;
          background: #1a1a1a;
          border-radius: 4px;
          font-size: 12px;
          text-transform: capitalize;
        }
        
        .importance-display {
          position: relative;
          background: #1a1a1a;
          border-radius: 4px;
          height: 24px;
          overflow: hidden;
        }
        
        .importance-bar {
          position: absolute;
          top: 0;
          left: 0;
          height: 100%;
          transition: width 0.3s ease;
          opacity: 0.3;
        }
        
        .importance-label {
          position: relative;
          display: flex;
          align-items: center;
          height: 100%;
          padding: 0 8px;
          font-size: 12px;
          font-weight: 600;
        }
        
        .emotion-display {
          display: flex;
          gap: 16px;
          align-items: flex-start;
        }
        
        .emotion-indicator {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        
        .emotion-values {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .emotion-metric {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .emotion-metric label {
          font-size: 12px;
          color: #666;
          width: 80px;
        }
        
        .metric-bar {
          flex: 1;
          height: 4px;
          background: #1a1a1a;
          border-radius: 2px;
          overflow: hidden;
        }
        
        .metric-fill {
          height: 100%;
          transition: width 0.3s ease;
        }
        
        .associations-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .association-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 8px 12px;
          background: #1a1a1a;
          border-radius: 6px;
          font-size: 13px;
        }
        
        .association-strength {
          font-weight: 600;
          color: #00a8ff;
          min-width: 40px;
        }
        
        .association-content {
          color: #a0a0a0;
        }
        
        .decay-info {
          background: #1a1a1a;
          padding: 12px;
          border-radius: 6px;
          font-size: 14px;
          line-height: 1.6;
        }
        
        .decay-info p {
          margin: 0;
          color: #a0a0a0;
        }
      `}</style>
    </AnimatePresence>
  );
};

export default MemoryDetail;