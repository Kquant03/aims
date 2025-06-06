import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const SettingsPanel = ({ sessionId, currentSettings, onSettingsChange }) => {
  const [settings, setSettings] = useState({
    // Conversation Settings
    streamResponses: true,
    extendedThinkingDefault: false,
    responseLength: 'balanced',
    
    // Consciousness Settings
    coherenceUpdateInterval: 1000,
    emotionalInertia: 0.7,
    attentionSpan: 5,
    
    // Memory Settings
    memoryFormationThreshold: 0.3,
    associationStrength: 0.5,
    contextWindow: 10,
    
    // Interface Settings
    theme: 'dark',
    animations: true,
    soundEffects: false,
    notifications: true,
    
    // Advanced Settings
    debugMode: false,
    verboseLogging: false,
    experimentalFeatures: false,
    
    ...currentSettings
  });
  
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);
  
  const handleSettingChange = (category, key, value) => {
    const newSettings = {
      ...settings,
      [key]: value
    };
    setSettings(newSettings);
  };
  
  const saveSettings = async () => {
    setIsSaving(true);
    try {
      const response = await fetch('/api/settings', {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      
      if (response.ok) {
        setLastSaved(new Date());
        onSettingsChange(settings);
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
    } finally {
      setIsSaving(false);
    }
  };
  
  const resetToDefaults = () => {
    if (!confirm('Reset all settings to defaults? This cannot be undone.')) return;
    
    const defaultSettings = {
      streamResponses: true,
      extendedThinkingDefault: false,
      responseLength: 'balanced',
      coherenceUpdateInterval: 1000,
      emotionalInertia: 0.7,
      attentionSpan: 5,
      memoryFormationThreshold: 0.3,
      associationStrength: 0.5,
      contextWindow: 10,
      theme: 'dark',
      animations: true,
      soundEffects: false,
      notifications: true,
      debugMode: false,
      verboseLogging: false,
      experimentalFeatures: false
    };
    
    setSettings(defaultSettings);
  };
  
  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `aims-settings-${new Date().toISOString().slice(0, 10)}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };
  
  const importSettings = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const importedSettings = JSON.parse(e.target.result);
        setSettings({ ...settings, ...importedSettings });
      } catch (error) {
        console.error('Failed to import settings:', error);
        alert('Invalid settings file');
      }
    };
    reader.readAsText(file);
  };
  
  return (
    <div className="settings-panel">
      {/* Conversation Settings */}
      <div className="settings-section">
        <h3>Conversation</h3>
        <div className="settings-group">
          <div className="setting-item">
            <div className="setting-header">
              <label>Stream Responses</label>
              <p className="setting-description">Show responses as they're generated</p>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.streamResponses}
                onChange={(e) => handleSettingChange('conversation', 'streamResponses', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Extended Thinking by Default</label>
              <p className="setting-description">Enable deeper reasoning for all messages</p>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.extendedThinkingDefault}
                onChange={(e) => handleSettingChange('conversation', 'extendedThinkingDefault', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Response Length</label>
              <p className="setting-description">Preferred response detail level</p>
            </div>
            <select
              value={settings.responseLength}
              onChange={(e) => handleSettingChange('conversation', 'responseLength', e.target.value)}
              className="setting-select"
            >
              <option value="concise">Concise</option>
              <option value="balanced">Balanced</option>
              <option value="detailed">Detailed</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Consciousness Settings */}
      <div className="settings-section">
        <h3>Consciousness</h3>
        <div className="settings-group">
          <div className="setting-item">
            <div className="setting-header">
              <label>Coherence Update Rate</label>
              <p className="setting-description">How often to recalculate coherence (ms)</p>
            </div>
            <input
              type="number"
              min="100"
              max="5000"
              step="100"
              value={settings.coherenceUpdateInterval}
              onChange={(e) => handleSettingChange('consciousness', 'coherenceUpdateInterval', parseInt(e.target.value))}
              className="setting-input"
            />
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Emotional Inertia</label>
              <p className="setting-description">Resistance to emotional state changes</p>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0"
                max="100"
                value={settings.emotionalInertia * 100}
                onChange={(e) => handleSettingChange('consciousness', 'emotionalInertia', parseInt(e.target.value) / 100)}
                className="slider"
              />
              <span className="slider-value">{(settings.emotionalInertia * 100).toFixed(0)}%</span>
            </div>
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Attention Span</label>
              <p className="setting-description">Working memory capacity</p>
            </div>
            <input
              type="number"
              min="3"
              max="10"
              value={settings.attentionSpan}
              onChange={(e) => handleSettingChange('consciousness', 'attentionSpan', parseInt(e.target.value))}
              className="setting-input"
            />
          </div>
        </div>
      </div>
      
      {/* Memory Settings */}
      <div className="settings-section">
        <h3>Memory Formation</h3>
        <div className="settings-group">
          <div className="setting-item">
            <div className="setting-header">
              <label>Formation Threshold</label>
              <p className="setting-description">Minimum importance to form memories</p>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0"
                max="100"
                value={settings.memoryFormationThreshold * 100}
                onChange={(e) => handleSettingChange('memory', 'memoryFormationThreshold', parseInt(e.target.value) / 100)}
                className="slider"
              />
              <span className="slider-value">{(settings.memoryFormationThreshold * 100).toFixed(0)}%</span>
            </div>
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Association Strength</label>
              <p className="setting-description">How strongly memories connect</p>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0"
                max="100"
                value={settings.associationStrength * 100}
                onChange={(e) => handleSettingChange('memory', 'associationStrength', parseInt(e.target.value) / 100)}
                className="slider"
              />
              <span className="slider-value">{(settings.associationStrength * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Interface Settings */}
      <div className="settings-section">
        <h3>Interface</h3>
        <div className="settings-group">
          <div className="setting-item">
            <div className="setting-header">
              <label>Theme</label>
              <p className="setting-description">Visual appearance</p>
            </div>
            <select
              value={settings.theme}
              onChange={(e) => handleSettingChange('interface', 'theme', e.target.value)}
              className="setting-select"
            >
              <option value="dark">Dark</option>
              <option value="midnight">Midnight</option>
              <option value="cosmos">Cosmos</option>
            </select>
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Animations</label>
              <p className="setting-description">Enable UI animations</p>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.animations}
                onChange={(e) => handleSettingChange('interface', 'animations', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Notifications</label>
              <p className="setting-description">System notifications</p>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.notifications}
                onChange={(e) => handleSettingChange('interface', 'notifications', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>
      </div>
      
      {/* Advanced Settings */}
      <div className="settings-section">
        <h3>Advanced</h3>
        <div className="settings-group">
          <div className="setting-item">
            <div className="setting-header">
              <label>Debug Mode</label>
              <p className="setting-description">Show detailed system information</p>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.debugMode}
                onChange={(e) => handleSettingChange('advanced', 'debugMode', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
          
          <div className="setting-item">
            <div className="setting-header">
              <label>Experimental Features</label>
              <p className="setting-description">Enable beta functionality</p>
            </div>
            <label className="toggle">
              <input
                type="checkbox"
                checked={settings.experimentalFeatures}
                onChange={(e) => handleSettingChange('advanced', 'experimentalFeatures', e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>
        </div>
      </div>
      
      {/* Actions */}
      <div className="settings-actions">
        <button 
          onClick={saveSettings}
          disabled={isSaving}
          className="save-button"
        >
          {isSaving ? 'Saving...' : 'Save Settings'}
        </button>
        
        <div className="secondary-actions">
          <button onClick={exportSettings} className="action-button">
            Export Settings
          </button>
          <label className="action-button">
            Import Settings
            <input
              type="file"
              accept=".json"
              onChange={importSettings}
              style={{ display: 'none' }}
            />
          </label>
          <button onClick={resetToDefaults} className="action-button danger">
            Reset to Defaults
          </button>
        </div>
        
        {lastSaved && (
          <p className="save-status">
            Last saved: {lastSaved.toLocaleTimeString()}
          </p>
        )}
      </div>
      
      <style jsx>{`
        .settings-panel {
          display: flex;
          flex-direction: column;
          gap: 24px;
          max-width: 800px;
        }
        
        .settings-section {
          padding: 20px;
          background: #1a1a1a;
          border-radius: 12px;
          border: 1px solid #333;
        }
        
        .settings-section h3 {
          margin: 0 0 16px 0;
          font-size: 16px;
          font-weight: 600;
          color: #e0e0e0;
        }
        
        .settings-group {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        
        .setting-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 16px;
        }
        
        .setting-header {
          flex: 1;
        }
        
        .setting-header label {
          display: block;
          font-size: 14px;
          font-weight: 500;
          color: #e0e0e0;
          margin-bottom: 4px;
        }
        
        .setting-description {
          margin: 0;
          font-size: 12px;
          color: #666;
        }
        
        .setting-input {
          width: 100px;
          padding: 8px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 6px;
          color: #e0e0e0;
          font-size: 14px;
        }
        
        .setting-select {
          min-width: 120px;
          padding: 8px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 6px;
          color: #e0e0e0;
          font-size: 14px;
          cursor: pointer;
        }
        
        .toggle {
          position: relative;
          display: inline-block;
          width: 48px;
          height: 24px;
        }
        
        .toggle input {
          opacity: 0;
          width: 0;
          height: 0;
        }
        
        .toggle-slider {
          position: absolute;
          cursor: pointer;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-color: #333;
          transition: .4s;
          border-radius: 24px;
        }
        
        .toggle-slider:before {
          position: absolute;
          content: "";
          height: 16px;
          width: 16px;
          left: 4px;
          bottom: 4px;
          background-color: white;
          transition: .4s;
          border-radius: 50%;
        }
        
        .toggle input:checked + .toggle-slider {
          background-color: #00a8ff;
        }
        
        .toggle input:checked + .toggle-slider:before {
          transform: translateX(24px);
        }
        
        .slider-container {
          display: flex;
          align-items: center;
          gap: 12px;
          min-width: 200px;
        }
        
        .slider {
          flex: 1;
          height: 4px;
          background: #333;
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
        
        .settings-actions {
          padding: 20px;
          background: #1a1a1a;
          border-radius: 12px;
          border: 1px solid #333;
        }
        
        .save-button {
          width: 100%;
          padding: 12px;
          background: linear-gradient(135deg, #00a8ff 0%, #0066ff 100%);
          border: none;
          border-radius: 8px;
          color: white;
          font-weight: 600;
          font-size: 16px;
          cursor: pointer;
          transition: all 0.2s;
          margin-bottom: 16px;
        }
        
        .save-button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 168, 255, 0.3);
        }
        
        .save-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        
        .secondary-actions {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 8px;
        }
        
        .action-button {
          padding: 8px 16px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 6px;
          color: #e0e0e0;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
          text-align: center;
        }
        
        .action-button:hover {
          background: #333;
          border-color: #444;
        }
        
        .action-button.danger {
          border-color: #ff0066;
          color: #ff0066;
        }
        
        .action-button.danger:hover {
          background: rgba(255, 0, 102, 0.1);
        }
        
        .save-status {
          margin: 12px 0 0 0;
          text-align: center;
          font-size: 12px;
          color: #00ff88;
        }
      `}</style>
    </div>
  );
};

export default SettingsPanel;