import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const StateManager = ({ sessionId, onStateLoaded }) => {
  const [savedStates, setSavedStates] = useState([]);
  const [selectedState, setSelectedState] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [newStateName, setNewStateName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  useEffect(() => {
    fetchSavedStates();
  }, []);
  
  const fetchSavedStates = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/state/list', {
        credentials: 'include'
      });
      const data = await response.json();
      setSavedStates(data);
    } catch (error) {
      console.error('Failed to fetch states:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const saveCurrentState = async () => {
    if (!newStateName.trim()) {
      alert('Please enter a name for the state');
      return;
    }
    
    setIsSaving(true);
    try {
      const response = await fetch('/api/state/save', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newStateName,
          description: `Saved on ${new Date().toLocaleString()}`
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        await fetchSavedStates();
        setShowSaveDialog(false);
        setNewStateName('');
      }
    } catch (error) {
      console.error('Failed to save state:', error);
    } finally {
      setIsSaving(false);
    }
  };
  
  const loadState = async (stateId) => {
    if (!confirm('Loading this state will replace your current consciousness state. Continue?')) {
      return;
    }
    
    try {
      const response = await fetch('/api/state/load', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state_id: stateId })
      });
      
      if (response.ok) {
        const loadedState = await response.json();
        onStateLoaded(loadedState);
        setSelectedState(stateId);
      }
    } catch (error) {
      console.error('Failed to load state:', error);
    }
  };
  
  const deleteState = async (stateId) => {
    if (!confirm('Are you sure you want to delete this state? This cannot be undone.')) {
      return;
    }
    
    try {
      await fetch(`/api/state/${stateId}`, {
        method: 'DELETE',
        credentials: 'include'
      });
      await fetchSavedStates();
    } catch (error) {
      console.error('Failed to delete state:', error);
    }
  };
  
  const exportState = async (state) => {
    try {
      const response = await fetch(`/api/state/${state.id}/export`, {
        credentials: 'include'
      });
      const blob = await response.blob();
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `aims-state-${state.name}-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Failed to export state:', error);
    }
  };
  
  const filteredStates = savedStates.filter(state => 
    state.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    state.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffDays === 0) {
      return `Today at ${date.toLocaleTimeString()}`;
    } else if (diffDays === 1) {
      return `Yesterday at ${date.toLocaleTimeString()}`;
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };
  
  return (
    <div className="state-manager">
      <div className="manager-header">
        <h3>Consciousness States</h3>
        <button 
          onClick={() => setShowSaveDialog(true)}
          className="save-state-button"
        >
          <span className="icon">💾</span>
          Save Current State
        </button>
      </div>
      
      {savedStates.length > 0 && (
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search saved states..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>
      )}
      
      {isLoading ? (
        <div className="loading-state">
          <div className="loader" />
          <p>Loading saved states...</p>
        </div>
      ) : filteredStates.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🗄️</div>
          <p>No saved states found</p>
          <span>Save your current consciousness state to preserve memories, personality, and emotional context</span>
        </div>
      ) : (
        <div className="states-grid">
          {filteredStates.map(state => (
            <motion.div
              key={state.id}
              className={`state-card ${selectedState === state.id ? 'selected' : ''}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="state-header">
                <h4>{state.name}</h4>
                <div className="state-actions">
                  <button
                    onClick={() => exportState(state)}
                    className="action-button"
                    title="Export state"
                  >
                    📤
                  </button>
                  <button
                    onClick={() => deleteState(state.id)}
                    className="action-button danger"
                    title="Delete state"
                  >
                    🗑️
                  </button>
                </div>
              </div>
              
              <div className="state-info">
                <p className="state-description">{state.description}</p>
                <div className="state-meta">
                  <span className="meta-item">
                    <span className="label">Saved:</span> {formatTimestamp(state.timestamp)}
                  </span>
                  <span className="meta-item">
                    <span className="label">Size:</span> {(state.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>
              
              <div className="state-preview">
                {state.preview && (
                  <>
                    <div className="preview-item">
                      <span className="preview-label">Coherence:</span>
                      <span className="preview-value">{(state.preview.coherence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="preview-item">
                      <span className="preview-label">Memories:</span>
                      <span className="preview-value">{state.preview.memory_count}</span>
                    </div>
                    <div className="preview-item">
                      <span className="preview-label">Emotion:</span>
                      <span className="preview-value">{state.preview.emotional_state}</span>
                    </div>
                  </>
                )}
              </div>
              
              <button 
                onClick={() => loadState(state.id)}
                className="load-button"
              >
                Load This State
              </button>
            </motion.div>
          ))}
        </div>
      )}
      
      <AnimatePresence>
        {showSaveDialog && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="dialog-overlay"
            onClick={() => setShowSaveDialog(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="save-dialog"
              onClick={(e) => e.stopPropagation()}
            >
              <h3>Save Current State</h3>
              <p>Capture your current consciousness state including memories, personality, and emotional context.</p>
              
              <input
                type="text"
                placeholder="Enter a name for this state..."
                value={newStateName}
                onChange={(e) => setNewStateName(e.target.value)}
                className="state-name-input"
                autoFocus
              />
              
              <div className="dialog-actions">
                <button 
                  onClick={saveCurrentState}
                  disabled={!newStateName.trim() || isSaving}
                  className="save-button"
                >
                  {isSaving ? 'Saving...' : 'Save State'}
                </button>
                <button 
                  onClick={() => {
                    setShowSaveDialog(false);
                    setNewStateName('');
                  }}
                  className="cancel-button"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .state-manager {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        
        .manager-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .manager-header h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }
        
        .save-state-button {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 16px;
          background: linear-gradient(135deg, #00a8ff 0%, #0066ff 100%);
          border: none;
          border-radius: 8px;
          color: white;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .save-state-button:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 168, 255, 0.3);
        }
        
        .save-state-button .icon {
          font-size: 18px;
        }
        
        .search-bar {
          position: relative;
        }
        
        .search-input {
          width: 100%;
          padding: 12px 16px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          color: #e0e0e0;
          font-size: 14px;
          transition: all 0.2s;
        }
        
        .search-input:focus {
          outline: none;
          border-color: #00a8ff;
          box-shadow: 0 0 0 2px rgba(0, 168, 255, 0.2);
        }
        
        .loading-state, .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 300px;
          text-align: center;
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
        
        .empty-icon {
          font-size: 48px;
          margin-bottom: 16px;
          opacity: 0.5;
        }
        
        .empty-state p {
          margin: 0 0 8px 0;
          font-size: 16px;
          font-weight: 600;
          color: #888;
        }
        
        .empty-state span {
          font-size: 14px;
          color: #666;
          max-width: 300px;
        }
        
        .states-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 16px;
        }
        
        .state-card {
          padding: 20px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 12px;
          cursor: default;
          transition: all 0.2s;
        }
        
        .state-card:hover {
          border-color: #444;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .state-card.selected {
          border-color: #00a8ff;
          box-shadow: 0 0 0 2px rgba(0, 168, 255, 0.2);
        }
        
        .state-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 12px;
        }
        
        .state-header h4 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
          color: #e0e0e0;
        }
        
        .state-actions {
          display: flex;
          gap: 4px;
        }
        
        .action-button {
          padding: 4px;
          background: none;
          border: none;
          font-size: 16px;
          cursor: pointer;
          opacity: 0.6;
          transition: opacity 0.2s;
        }
        
        .action-button:hover {
          opacity: 1;
        }
        
        .action-button.danger:hover {
          color: #ff0066;
        }
        
        .state-info {
          margin-bottom: 16px;
        }
        
        .state-description {
          margin: 0 0 8px 0;
          font-size: 13px;
          color: #888;
        }
        
        .state-meta {
          display: flex;
          gap: 16px;
          font-size: 12px;
          color: #666;
        }
        
        .meta-item .label {
          font-weight: 600;
        }
        
        .state-preview {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 12px;
          padding: 12px;
          background: #0a0a0a;
          border-radius: 8px;
          margin-bottom: 16px;
        }
        
        .preview-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
        }
        
        .preview-label {
          font-size: 11px;
          color: #666;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .preview-value {
          font-size: 14px;
          font-weight: 600;
          color: #00a8ff;
        }
        
        .load-button {
          width: 100%;
          padding: 10px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 8px;
          color: #e0e0e0;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .load-button:hover {
          background: #333;
          border-color: #00a8ff;
          color: #00a8ff;
        }
        
        .dialog-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.7);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }
        
        .save-dialog {
          width: 90%;
          max-width: 480px;
          padding: 24px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 12px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        
        .save-dialog h3 {
          margin: 0 0 12px 0;
          font-size: 20px;
          font-weight: 600;
        }
        
        .save-dialog p {
          margin: 0 0 20px 0;
          font-size: 14px;
          color: #888;
          line-height: 1.5;
        }
        
        .state-name-input {
          width: 100%;
          padding: 12px 16px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 8px;
          color: #e0e0e0;
          font-size: 16px;
          margin-bottom: 20px;
        }
        
        .state-name-input:focus {
          outline: none;
          border-color: #00a8ff;
          box-shadow: 0 0 0 2px rgba(0, 168, 255, 0.2);
        }
        
        .dialog-actions {
          display: flex;
          gap: 12px;
        }
        
        .save-button {
          flex: 1;
          padding: 12px;
          background: linear-gradient(135deg, #00a8ff 0%, #0066ff 100%);
          border: none;
          border-radius: 8px;
          color: white;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .save-button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 168, 255, 0.3);
        }
        
        .save-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        .cancel-button {
          padding: 12px 24px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 8px;
          color: #a0a0a0;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .cancel-button:hover {
          background: #333;
          color: #e0e0e0;
        }
      `}</style>
    </div>
  );
};

export default StateManager;