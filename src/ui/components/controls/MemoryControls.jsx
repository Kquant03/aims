import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const MemoryControls = ({ sessionId, memoryStats: initialStats }) => {
  const [memoryStats, setMemoryStats] = useState(initialStats || null);
  const [consolidationStatus, setConsolidationStatus] = useState(null);
  const [isConsolidating, setIsConsolidating] = useState(false);
  const [memorySettings, setMemorySettings] = useState({
    autoConsolidate: true,
    consolidationInterval: 300, // 5 minutes
    decayEnabled: true,
    decayRate: 0.1,
    importanceThreshold: 0.3
  });
  
  useEffect(() => {
    fetchMemoryStats();
    fetchConsolidationStatus();
  }, [sessionId]);
  
  const fetchMemoryStats = async () => {
    try {
      const response = await fetch('/api/memory/stats', {
        credentials: 'include'
      });
      const data = await response.json();
      setMemoryStats(data);
    } catch (error) {
      console.error('Failed to fetch memory stats:', error);
    }
  };
  
  const fetchConsolidationStatus = async () => {
    try {
      const response = await fetch('/api/memory/consolidation/status', {
        credentials: 'include'
      });
      const data = await response.json();
      setConsolidationStatus(data);
    } catch (error) {
      console.error('Failed to fetch consolidation status:', error);
    }
  };
  
  const triggerConsolidation = async () => {
    setIsConsolidating(true);
    try {
      const response = await fetch('/api/memory/consolidate', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      
      if (response.ok) {
        const result = await response.json();
        setConsolidationStatus(result);
        // Refresh stats after consolidation
        setTimeout(fetchMemoryStats, 2000);
      }
    } catch (error) {
      console.error('Consolidation failed:', error);
    } finally {
      setIsConsolidating(false);
    }
  };
  
  const updateMemorySettings = async (newSettings) => {
    try {
      const response = await fetch('/api/memory/settings', {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSettings)
      });
      
      if (response.ok) {
        setMemorySettings(newSettings);
      }
    } catch (error) {
      console.error('Failed to update memory settings:', error);
    }
  };
  
  const clearMemoryTier = async (tier) => {
    const confirmMessage = {
      working: 'Clear all working memory? This cannot be undone.',
      episodic: 'Clear all episodic memories? This will remove conversation history.',
      semantic: 'Clear all semantic knowledge? This will reset learned concepts.'
    };
    
    if (!confirm(confirmMessage[tier])) return;
    
    try {
      await fetch(`/api/memory/clear/${tier}`, {
        method: 'POST',
        credentials: 'include'
      });
      fetchMemoryStats();
    } catch (error) {
      console.error(`Failed to clear ${tier} memory:`, error);
    }
  };
  
  const getMemoryUsagePercentage = (used, total) => {
    if (!total) return 0;
    return Math.min((used / total) * 100, 100);
  };
  
  if (!memoryStats) {
    return (
      <div className="memory-controls loading">
        <div className="loader" />
        <p>Loading memory controls...</p>
      </div>
    );
  }
  
  return (
    <div className="memory-controls">
      <div className="controls-section">
        <h3>Memory Overview</h3>
        
        <div className="memory-tiers">
          {/* Working Memory */}
          <div className="memory-tier">
            <div className="tier-header">
              <div className="tier-info">
                <h4>Working Memory</h4>
                <span className="tier-description">Active thoughts and current context</span>
              </div>
              <button 
                onClick={() => clearMemoryTier('working')}
                className="clear-button"
                title="Clear working memory"
              >
                🗑️
              </button>
            </div>
            
            <div className="tier-stats">
              <div className="stat-item">
                <label>Items</label>
                <span>{memoryStats.working_memory_count || 0}</span>
              </div>
              <div className="stat-item">
                <label>Capacity</label>
                <div className="capacity-bar">
                  <div 
                    className="capacity-fill working"
                    style={{ 
                      width: `${getMemoryUsagePercentage(
                        memoryStats.working_memory_count || 0, 
                        memoryStats.working_memory_limit || 10
                      )}%` 
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
          
          {/* Episodic Memory */}
          <div className="memory-tier">
            <div className="tier-header">
              <div className="tier-info">
                <h4>Episodic Memory</h4>
                <span className="tier-description">Conversation history and experiences</span>
              </div>
              <button 
                onClick={() => clearMemoryTier('episodic')}
                className="clear-button"
                title="Clear episodic memory"
              >
                🗑️
              </button>
            </div>
            
            <div className="tier-stats">
              <div className="stat-item">
                <label>Memories</label>
                <span>{memoryStats.episodic_count || 0}</span>
              </div>
              <div className="stat-item">
                <label>Avg Importance</label>
                <span>{((memoryStats.avg_importance || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
          
          {/* Semantic Memory */}
          <div className="memory-tier">
            <div className="tier-header">
              <div className="tier-info">
                <h4>Semantic Memory</h4>
                <span className="tier-description">Consolidated knowledge and concepts</span>
              </div>
              <button 
                onClick={() => clearMemoryTier('semantic')}
                className="clear-button"
                title="Clear semantic memory"
              >
                🗑️
              </button>
            </div>
            
            <div className="tier-stats">
              <div className="stat-item">
                <label>Concepts</label>
                <span>{memoryStats.semantic_count || 0}</span>
              </div>
              <div className="stat-item">
                <label>Clusters</label>
                <span>{memoryStats.cluster_count || 0}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="controls-section">
        <h3>Memory Consolidation</h3>
        
        <div className="consolidation-status">
          {consolidationStatus ? (
            <>
              <div className="status-item">
                <label>Last Run</label>
                <span>{new Date(consolidationStatus.last_run).toLocaleString()}</span>
              </div>
              <div className="status-item">
                <label>Memories Processed</label>
                <span>{consolidationStatus.memories_processed || 0}</span>
              </div>
              <div className="status-item">
                <label>New Concepts</label>
                <span>{consolidationStatus.new_concepts || 0}</span>
              </div>
            </>
          ) : (
            <p className="no-data">No consolidation data available</p>
          )}
        </div>
        
        <button 
          onClick={triggerConsolidation}
          disabled={isConsolidating}
          className="consolidate-button"
        >
          {isConsolidating ? (
            <>
              <span className="spinner" />
              Consolidating...
            </>
          ) : (
            <>
              <span className="icon">🧬</span>
              Consolidate Now
            </>
          )}
        </button>
      </div>
      
      <div className="controls-section">
        <h3>Memory Settings</h3>
        
        <div className="settings-grid">
          <div className="setting-item">
            <label>
              <input 
                type="checkbox"
                checked={memorySettings.autoConsolidate}
                onChange={(e) => updateMemorySettings({
                  ...memorySettings,
                  autoConsolidate: e.target.checked
                })}
              />
              <span>Auto-consolidate memories</span>
            </label>
          </div>
          
          <div className="setting-item">
            <label>Consolidation Interval</label>
            <select 
              value={memorySettings.consolidationInterval}
              onChange={(e) => updateMemorySettings({
                ...memorySettings,
                consolidationInterval: parseInt(e.target.value)
              })}
            >
              <option value="60">1 minute</option>
              <option value="300">5 minutes</option>
              <option value="900">15 minutes</option>
              <option value="3600">1 hour</option>
            </select>
          </div>
          
          <div className="setting-item">
            <label>
              <input 
                type="checkbox"
                checked={memorySettings.decayEnabled}
                onChange={(e) => updateMemorySettings({
                  ...memorySettings,
                  decayEnabled: e.target.checked
                })}
              />
              <span>Enable memory decay</span>
            </label>
          </div>
          
          <div className="setting-item">
            <label>Importance Threshold</label>
            <div className="slider-container">
              <input 
                type="range"
                min="0"
                max="100"
                value={memorySettings.importanceThreshold * 100}
                onChange={(e) => updateMemorySettings({
                  ...memorySettings,
                  importanceThreshold: parseInt(e.target.value) / 100
                })}
                className="slider"
              />
              <span className="slider-value">
                {(memorySettings.importanceThreshold * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .memory-controls {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }
        
        .memory-controls.loading {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 300px;
          color: #666;
        }
        
        .loader {
          width: 32px;
          height: 32px;
          border: 3px solid #333;
          border-top-color: #00a8ff;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-bottom: 16px;
        }
        
        .controls-section {
          padding: 20px;
          background: #1a1a1a;
          border-radius: 12px;
          border: 1px solid #333;
        }
        
        .controls-section h3 {
          margin: 0 0 16px 0;
          font-size: 16px;
          font-weight: 600;
        }
        
        .memory-tiers {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        
        .memory-tier {
          padding: 16px;
          background: #0a0a0a;
          border: 1px solid #252525;
          border-radius: 8px;
        }
        
        .tier-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 12px;
        }
        
        .tier-info h4 {
          margin: 0 0 4px 0;
          font-size: 14px;
          font-weight: 600;
        }
        
        .tier-description {
          font-size: 12px;
          color: #666;
        }
        
        .clear-button {
          padding: 6px;
          background: none;
          border: none;
          font-size: 16px;
          cursor: pointer;
          opacity: 0.5;
          transition: opacity 0.2s;
        }
        
        .clear-button:hover {
          opacity: 1;
        }
        
        .tier-stats {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 12px;
        }
        
        .stat-item {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        
        .stat-item label {
          font-size: 11px;
          color: #666;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .stat-item span {
          font-size: 18px;
          font-weight: 600;
          color: #e0e0e0;
        }
        
        .capacity-bar {
          width: 100%;
          height: 4px;
          background: #252525;
          border-radius: 2px;
          overflow: hidden;
        }
        
        .capacity-fill {
          height: 100%;
          transition: width 0.3s ease;
        }
        
        .capacity-fill.working {
          background: #00a8ff;
        }
        
        .consolidation-status {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 16px;
          margin-bottom: 16px;
          padding: 16px;
          background: #0a0a0a;
          border-radius: 8px;
        }
        
        .status-item {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        
        .status-item label {
          font-size: 12px;
          color: #666;
          font-weight: 600;
        }
        
        .status-item span {
          font-size: 14px;
          color: #e0e0e0;
        }
        
        .no-data {
          color: #666;
          font-style: italic;
          margin: 0;
        }
        
        .consolidate-button {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          width: 100%;
          padding: 12px;
          background: linear-gradient(135deg, #00a8ff 0%, #0066ff 100%);
          border: none;
          border-radius: 8px;
          color: white;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .consolidate-button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 168, 255, 0.3);
        }
        
        .consolidate-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        .consolidate-button .icon {
          font-size: 18px;
        }
        
        .consolidate-button .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        
        .settings-grid {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        
        .setting-item {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .setting-item label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #e0e0e0;
          cursor: pointer;
        }
        
        .setting-item input[type="checkbox"] {
          width: 16px;
          height: 16px;
          cursor: pointer;
        }
        
        .setting-item select {
          padding: 8px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 6px;
          color: #e0e0e0;
          font-size: 14px;
          cursor: pointer;
        }
        
        .slider-container {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .slider {
          flex: 1;
          height: 4px;
          background: #252525;
          border-radius: 2px;
          outline: none;
          -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px;
          height: 16px;
          background: #00a8ff;
          border-radius: 50%;
          cursor: pointer;
        }
        
        .slider-value {
          min-width: 40px;
          text-align: right;
          font-size: 14px;
          font-weight: 600;
          color: #00a8ff;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default MemoryControls;