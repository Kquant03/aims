import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import StateManager from './StateManager';
import PersonalityTuner from './PersonalityTuner';
import MemoryControls from './MemoryControls';
import SettingsPanel from './SettingsPanel';

const SystemControls = ({ sessionId, onStateChange }) => {
  const [activePanel, setActivePanel] = useState('state');
  const [systemStatus, setSystemStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    fetchSystemStatus();
  }, [sessionId]);
  
  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('/api/consciousness/state', {
        credentials: 'include'
      });
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const controlPanels = [
    { id: 'state', label: 'State Manager', icon: '💾' },
    { id: 'personality', label: 'Personality', icon: '🎭' },
    { id: 'memory', label: 'Memory', icon: '🧠' },
    { id: 'settings', label: 'Settings', icon: '⚙️' }
  ];
  
  const handleEmergencyReset = async () => {
    if (!confirm('Are you sure you want to perform an emergency reset? This will clear the current session but preserve saved states.')) {
      return;
    }
    
    try {
      const response = await fetch('/api/system/reset', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, preserve_states: true })
      });
      
      if (response.ok) {
        window.location.reload();
      }
    } catch (error) {
      console.error('Emergency reset failed:', error);
    }
  };
  
  const handleExportData = async () => {
    try {
      const response = await fetch('/api/system/export', {
        credentials: 'include'
      });
      const blob = await response.blob();
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `aims-export-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };
  
  if (isLoading) {
    return (
      <div className="system-controls loading">
        <div className="loader" />
        <p>Loading system controls...</p>
      </div>
    );
  }
  
  return (
    <div className="system-controls">
      <div className="controls-header">
        <h2>System Controls</h2>
        <div className="system-indicators">
          <div className="indicator">
            <span className="indicator-dot active" />
            <span className="indicator-label">System Active</span>
          </div>
          {systemStatus && (
            <>
              <div className="indicator">
                <span className="indicator-value">{systemStatus.interaction_count || 0}</span>
                <span className="indicator-label">Interactions</span>
              </div>
              <div className="indicator">
                <span className="indicator-value">{systemStatus.working_memory_size || 0}</span>
                <span className="indicator-label">Working Memory</span>
              </div>
            </>
          )}
        </div>
      </div>
      
      <div className="controls-navigation">
        {controlPanels.map(panel => (
          <button
            key={panel.id}
            className={`nav-button ${activePanel === panel.id ? 'active' : ''}`}
            onClick={() => setActivePanel(panel.id)}
          >
            <span className="nav-icon">{panel.icon}</span>
            <span className="nav-label">{panel.label}</span>
          </button>
        ))}
      </div>
      
      <div className="controls-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={activePanel}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
            className="control-panel"
          >
            {activePanel === 'state' && (
              <StateManager
                sessionId={sessionId}
                onStateLoaded={(state) => {
                  onStateChange(state);
                  fetchSystemStatus();
                }}
              />
            )}
            
            {activePanel === 'personality' && (
              <PersonalityTuner
                currentPersonality={systemStatus?.personality}
                onPersonalityChange={(updates) => {
                  onStateChange({ personality: updates });
                }}
              />
            )}
            
            {activePanel === 'memory' && (
              <MemoryControls
                sessionId={sessionId}
                memoryStats={systemStatus?.memory_stats}
              />
            )}
            
            {activePanel === 'settings' && (
              <SettingsPanel
                sessionId={sessionId}
                currentSettings={systemStatus?.settings}
                onSettingsChange={(settings) => {
                  onStateChange({ settings });
                }}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
      
      <div className="system-actions">
        <div className="action-group">
          <h3>Quick Actions</h3>
          <div className="quick-actions">
            <button 
              onClick={handleExportData}
              className="action-button export"
            >
              <span className="action-icon">📤</span>
              Export All Data
            </button>
            
            <button 
              onClick={() => {
                // Trigger consciousness snapshot
                fetch('/api/consciousness/snapshot', {
                  method: 'POST',
                  credentials: 'include'
                });
              }}
              className="action-button snapshot"
            >
              <span className="action-icon">📸</span>
              Take Snapshot
            </button>
            
            <button 
              onClick={() => {
                // Clear working memory
                if (confirm('Clear working memory? This will not affect long-term memories.')) {
                  fetch('/api/memory/clear-working', {
                    method: 'POST',
                    credentials: 'include'
                  });
                }
              }}
              className="action-button clear"
            >
              <span className="action-icon">🧹</span>
              Clear Working Memory
            </button>
          </div>
        </div>
        
        <div className="danger-zone">
          <h3>Danger Zone</h3>
          <p>These actions can significantly affect the system state.</p>
          <button 
            onClick={handleEmergencyReset}
            className="danger-button"
          >
            Emergency Reset
          </button>
        </div>
      </div>
      
      <style jsx>{`
        .system-controls {
          display: flex;
          flex-direction: column;
          height: 100%;
          background: #0a0a0a;
          border-radius: 12px;
          overflow: hidden;
        }
        
        .system-controls.loading {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 400px;
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
        
        .controls-header {
          padding: 24px;
          border-bottom: 1px solid #333;
        }
        
        .controls-header h2 {
          margin: 0 0 16px 0;
          font-size: 24px;
          font-weight: 600;
        }
        
        .system-indicators {
          display: flex;
          gap: 24px;
        }
        
        .indicator {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .indicator-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #666;
        }
        
        .indicator-dot.active {
          background: #00ff88;
          box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
        }
        
        .indicator-label {
          font-size: 13px;
          color: #888;
        }
        
        .indicator-value {
          font-size: 16px;
          font-weight: 600;
          color: #e0e0e0;
        }
        
        .controls-navigation {
          display: flex;
          gap: 2px;
          padding: 16px;
          background: #1a1a1a;
          border-bottom: 1px solid #333;
        }
        
        .nav-button {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 12px;
          background: transparent;
          border: 1px solid transparent;
          border-radius: 8px;
          color: #666;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .nav-button:hover {
          background: #252525;
          color: #e0e0e0;
        }
        
        .nav-button.active {
          background: #252525;
          border-color: #333;
          color: #e0e0e0;
        }
        
        .nav-icon {
          font-size: 20px;
        }
        
        .nav-label {
          font-size: 14px;
          font-weight: 500;
        }
        
        .controls-content {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
        }
        
        .control-panel {
          min-height: 400px;
        }
        
        .system-actions {
          padding: 24px;
          background: #1a1a1a;
          border-top: 1px solid #333;
        }
        
        .action-group {
          margin-bottom: 24px;
        }
        
        .action-group h3 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .quick-actions {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
          gap: 12px;
        }
        
        .action-button {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 8px;
          color: #e0e0e0;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .action-button:hover {
          background: #333;
          border-color: #444;
          transform: translateY(-1px);
        }
        
        .action-button.export:hover {
          border-color: #00a8ff;
        }
        
        .action-button.snapshot:hover {
          border-color: #00ff88;
        }
        
        .action-button.clear:hover {
          border-color: #ffaa00;
        }
        
        .action-icon {
          font-size: 18px;
        }
        
        .danger-zone {
          padding: 16px;
          background: rgba(255, 0, 102, 0.1);
          border: 1px solid rgba(255, 0, 102, 0.3);
          border-radius: 8px;
        }
        
        .danger-zone h3 {
          margin: 0 0 8px 0;
          font-size: 14px;
          font-weight: 600;
          color: #ff0066;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .danger-zone p {
          margin: 0 0 12px 0;
          font-size: 13px;
          color: #a0a0a0;
        }
        
        .danger-button {
          padding: 8px 16px;
          background: rgba(255, 0, 102, 0.2);
          border: 1px solid #ff0066;
          border-radius: 6px;
          color: #ff0066;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .danger-button:hover {
          background: #ff0066;
          color: white;
        }
        
        @media (max-width: 768px) {
          .controls-navigation {
            flex-wrap: wrap;
          }
          
          .nav-label {
            display: none;
          }
        }
      `}</style>
    </div>
  );
};

export default SystemControls;