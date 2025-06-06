import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './PersonalityTuner.css';

const PersonalityTuner = ({ currentTraits = {}, onTraitChange, onReset }) => {
  const [localTraits, setLocalTraits] = useState({
    openness: 0.8,
    conscientiousness: 0.7,
    extraversion: 0.6,
    agreeableness: 0.8,
    neuroticism: 0.3,
    ...currentTraits
  });
  
  const [editMode, setEditMode] = useState(false);
  const [presets, setPresets] = useState([
    {
      name: 'Balanced',
      traits: { openness: 0.7, conscientiousness: 0.7, extraversion: 0.6, agreeableness: 0.7, neuroticism: 0.4 }
    },
    {
      name: 'Creative',
      traits: { openness: 0.9, conscientiousness: 0.5, extraversion: 0.7, agreeableness: 0.6, neuroticism: 0.5 }
    },
    {
      name: 'Analytical',
      traits: { openness: 0.6, conscientiousness: 0.9, extraversion: 0.4, agreeableness: 0.6, neuroticism: 0.3 }
    },
    {
      name: 'Social',
      traits: { openness: 0.7, conscientiousness: 0.6, extraversion: 0.9, agreeableness: 0.8, neuroticism: 0.3 }
    },
    {
      name: 'Empathetic',
      traits: { openness: 0.8, conscientiousness: 0.7, extraversion: 0.6, agreeableness: 0.95, neuroticism: 0.4 }
    }
  ]);
  
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Trait information
  const traitInfo = {
    openness: {
      label: 'Openness',
      description: 'Creativity, curiosity, and openness to new experiences',
      low: 'Traditional, practical',
      high: 'Creative, imaginative',
      color: '#00a8ff',
      icon: '✨'
    },
    conscientiousness: {
      label: 'Conscientiousness',
      description: 'Organization, dependability, and self-discipline',
      low: 'Flexible, spontaneous',
      high: 'Organized, reliable',
      color: '#00ff88',
      icon: '📋'
    },
    extraversion: {
      label: 'Extraversion',
      description: 'Sociability, assertiveness, and emotional expression',
      low: 'Reserved, thoughtful',
      high: 'Outgoing, energetic',
      color: '#ffaa00',
      icon: '🗣️'
    },
    agreeableness: {
      label: 'Agreeableness',
      description: 'Cooperation, trust, and empathy',
      low: 'Competitive, skeptical',
      high: 'Cooperative, trusting',
      color: '#ff00aa',
      icon: '❤️'
    },
    neuroticism: {
      label: 'Neuroticism',
      description: 'Emotional instability, anxiety, and moodiness',
      low: 'Stable, calm',
      high: 'Sensitive, reactive',
      color: '#ff0066',
      icon: '🌊'
    }
  };
  
  // Update local traits when props change
  useEffect(() => {
    if (currentTraits) {
      setLocalTraits(prev => ({ ...prev, ...currentTraits }));
    }
  }, [currentTraits]);
  
  const handleTraitChange = (trait, value) => {
    const newTraits = { ...localTraits, [trait]: value };
    setLocalTraits(newTraits);
    
    if (editMode && onTraitChange) {
      onTraitChange(newTraits);
    }
  };
  
  const handlePresetSelect = (preset) => {
    setSelectedPreset(preset.name);
    setLocalTraits(preset.traits);
    
    if (onTraitChange) {
      onTraitChange(preset.traits);
    }
  };
  
  const calculatePersonalityProfile = () => {
    const profiles = [];
    
    if (localTraits.openness > 0.7 && localTraits.extraversion > 0.6) {
      profiles.push('Explorer');
    }
    if (localTraits.conscientiousness > 0.7 && localTraits.agreeableness > 0.7) {
      profiles.push('Caregiver');
    }
    if (localTraits.openness > 0.7 && localTraits.conscientiousness < 0.5) {
      profiles.push('Artist');
    }
    if (localTraits.extraversion < 0.4 && localTraits.conscientiousness > 0.7) {
      profiles.push('Analyst');
    }
    if (localTraits.agreeableness > 0.8 && localTraits.neuroticism < 0.3) {
      profiles.push('Peacemaker');
    }
    
    return profiles.length > 0 ? profiles.join(' / ') : 'Balanced';
  };
  
  const getTraitModifiers = () => {
    const modifiers = [];
    
    if (localTraits.openness > 0.8) {
      modifiers.push({ text: 'Highly creative responses', type: 'positive' });
    }
    if (localTraits.conscientiousness > 0.8) {
      modifiers.push({ text: 'Detailed and thorough', type: 'positive' });
    }
    if (localTraits.extraversion > 0.8) {
      modifiers.push({ text: 'Enthusiastic communication', type: 'positive' });
    }
    if (localTraits.agreeableness > 0.8) {
      modifiers.push({ text: 'Empathetic understanding', type: 'positive' });
    }
    if (localTraits.neuroticism > 0.6) {
      modifiers.push({ text: 'Heightened sensitivity', type: 'caution' });
    }
    
    return modifiers;
  };
  
  return (
    <div className="personality-tuner">
      <div className="tuner-header">
        <h3>Personality Configuration</h3>
        <div className="tuner-controls">
          <button
            className={`mode-toggle ${editMode ? 'active' : ''}`}
            onClick={() => setEditMode(!editMode)}
          >
            {editMode ? '🔓 Editing' : '🔒 Locked'}
          </button>
          <button
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '📊 Simple' : '🔬 Advanced'}
          </button>
        </div>
      </div>
      
      <div className="personality-profile">
        <div className="profile-label">Current Profile</div>
        <div className="profile-name">{calculatePersonalityProfile()}</div>
      </div>
      
      <div className="trait-sliders">
        {Object.entries(traitInfo).map(([trait, info]) => (
          <div key={trait} className="trait-control">
            <div className="trait-header">
              <span className="trait-icon">{info.icon}</span>
              <span className="trait-label">{info.label}</span>
              <span className="trait-value">{(localTraits[trait] * 100).toFixed(0)}%</span>
            </div>
            
            <div className="trait-slider-container">
              <span className="slider-label low">{info.low}</span>
              <input
                type="range"
                min="0"
                max="100"
                value={localTraits[trait] * 100}
                onChange={(e) => handleTraitChange(trait, e.target.value / 100)}
                disabled={!editMode}
                className="trait-slider"
                style={{
                  '--slider-color': info.color,
                  '--slider-value': `${localTraits[trait] * 100}%`
                }}
              />
              <span className="slider-label high">{info.high}</span>
            </div>
            
            {showAdvanced && (
              <div className="trait-description">{info.description}</div>
            )}
          </div>
        ))}
      </div>
      
      <div className="preset-section">
        <div className="section-header">Personality Presets</div>
        <div className="preset-grid">
          {presets.map((preset) => (
            <button
              key={preset.name}
              className={`preset-button ${selectedPreset === preset.name ? 'selected' : ''}`}
              onClick={() => handlePresetSelect(preset)}
              disabled={!editMode}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>
      
      <AnimatePresence>
        {showAdvanced && (
          <motion.div
            className="advanced-section"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="section-header">Behavioral Modifiers</div>
            <div className="modifier-list">
              {getTraitModifiers().map((modifier, index) => (
                <div key={index} className={`modifier-item ${modifier.type}`}>
                  <span className="modifier-icon">
                    {modifier.type === 'positive' ? '✓' : '!'}
                  </span>
                  <span className="modifier-text">{modifier.text}</span>
                </div>
              ))}
            </div>
            
            <div className="trait-visualization">
              <svg viewBox="0 0 300 300" className="personality-radar">
                <defs>
                  <radialGradient id="radarGradient">
                    <stop offset="0%" stopColor="#00ff88" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="#00ff88" stopOpacity="0.1" />
                  </radialGradient>
                </defs>
                
                {/* Background circles */}
                {[0.2, 0.4, 0.6, 0.8, 1].map((scale) => (
                  <circle
                    key={scale}
                    cx="150"
                    cy="150"
                    r={scale * 120}
                    fill="none"
                    stroke="#333"
                    strokeWidth="1"
                    opacity="0.3"
                  />
                ))}
                
                {/* Axis lines */}
                {Object.keys(traitInfo).map((trait, index) => {
                  const angle = (index * 72 - 90) * (Math.PI / 180);
                  const x = 150 + Math.cos(angle) * 120;
                  const y = 150 + Math.sin(angle) * 120;
                  return (
                    <line
                      key={trait}
                      x1="150"
                      y1="150"
                      x2={x}
                      y2={y}
                      stroke="#333"
                      strokeWidth="1"
                      opacity="0.3"
                    />
                  );
                })}
                
                {/* Personality shape */}
                <polygon
                  points={Object.keys(traitInfo).map((trait, index) => {
                    const angle = (index * 72 - 90) * (Math.PI / 180);
                    const radius = localTraits[trait] * 120;
                    const x = 150 + Math.cos(angle) * radius;
                    const y = 150 + Math.sin(angle) * radius;
                    return `${x},${y}`;
                  }).join(' ')}
                  fill="url(#radarGradient)"
                  stroke="#00ff88"
                  strokeWidth="2"
                />
                
                {/* Trait labels */}
                {Object.entries(traitInfo).map(([trait, info], index) => {
                  const angle = (index * 72 - 90) * (Math.PI / 180);
                  const x = 150 + Math.cos(angle) * 140;
                  const y = 150 + Math.sin(angle) * 140;
                  return (
                    <text
                      key={trait}
                      x={x}
                      y={y}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill="#888"
                      fontSize="12"
                    >
                      {info.icon} {info.label}
                    </text>
                  );
                })}
              </svg>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <div className="tuner-actions">
        <button
          className="reset-button"
          onClick={() => {
            if (onReset) onReset();
            setSelectedPreset(null);
          }}
          disabled={!editMode}
        >
          Reset to Defaults
        </button>
        <button
          className="save-button"
          onClick={() => onTraitChange && onTraitChange(localTraits)}
          disabled={!editMode}
        >
          Apply Changes
        </button>
      </div>
      
      <style jsx>{`
        .personality-tuner {
          background: rgba(26, 26, 26, 0.5);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 24px;
          max-width: 600px;
        }
        
        .tuner-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        
        .tuner-header h3 {
          margin: 0;
          color: #e0e0e0;
          font-size: 20px;
        }
        
        .tuner-controls {
          display: flex;
          gap: 8px;
        }
        
        .mode-toggle,
        .advanced-toggle {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          padding: 6px 12px;
          color: #888;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .mode-toggle:hover,
        .advanced-toggle:hover {
          background: rgba(255, 255, 255, 0.1);
          color: #e0e0e0;
        }
        
        .mode-toggle.active {
          background: rgba(0, 255, 136, 0.1);
          border-color: rgba(0, 255, 136, 0.3);
          color: #00ff88;
        }
        
        .personality-profile {
          background: rgba(0, 168, 255, 0.1);
          border: 1px solid rgba(0, 168, 255, 0.2);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 24px;
          text-align: center;
        }
        
        .profile-label {
          font-size: 11px;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 4px;
        }
        
        .profile-name {
          font-size: 18px;
          color: #00a8ff;
          font-weight: 600;
        }
        
        .trait-sliders {
          display: flex;
          flex-direction: column;
          gap: 20px;
          margin-bottom: 24px;
        }
        
        .trait-control {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .trait-header {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .trait-icon {
          font-size: 18px;
        }
        
        .trait-label {
          flex: 1;
          color: #e0e0e0;
          font-size: 14px;
          font-weight: 500;
        }
        
        .trait-value {
          color: #00ff88;
          font-size: 14px;
          font-weight: 600;
          min-width: 40px;
          text-align: right;
        }
        
        .trait-slider-container {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .slider-label {
          font-size: 11px;
          color: #666;
          min-width: 80px;
        }
        
        .slider-label.low {
          text-align: right;
        }
        
        .trait-slider {
          flex: 1;
          height: 6px;
          border-radius: 3px;
          outline: none;
          -webkit-appearance: none;
          background: linear-gradient(
            to right,
            var(--slider-color) 0%,
            var(--slider-color) var(--slider-value),
            rgba(255, 255, 255, 0.1) var(--slider-value),
            rgba(255, 255, 255, 0.1) 100%
          );
          cursor: pointer;
        }
        
        .trait-slider:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .trait-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--slider-color);
          cursor: pointer;
          border: 2px solid #0a0a0a;
          box-shadow: 0 0 0 1px var(--slider-color);
        }
        
        .trait-description {
          font-size: 12px;
          color: #666;
          padding-left: 26px;
        }
        
        .preset-section {
          margin-bottom: 24px;
        }
        
        .section-header {
          font-size: 12px;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 12px;
        }
        
        .preset-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
          gap: 8px;
        }
        
        .preset-button {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          padding: 8px 12px;
          color: #888;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .preset-button:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.1);
          color: #e0e0e0;
        }
        
        .preset-button.selected {
          background: rgba(0, 255, 136, 0.1);
          border-color: rgba(0, 255, 136, 0.3);
          color: #00ff88;
        }
        
        .preset-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .advanced-section {
          margin-bottom: 24px;
        }
        
        .modifier-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 20px;
        }
        
        .modifier-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          font-size: 13px;
        }
        
        .modifier-item.positive {
          border-color: rgba(0, 255, 136, 0.2);
          color: #00ff88;
        }
        
        .modifier-item.caution {
          border-color: rgba(255, 170, 0, 0.2);
          color: #ffaa00;
        }
        
        .modifier-icon {
          font-size: 14px;
        }
        
        .trait-visualization {
          display: flex;
          justify-content: center;
          margin-top: 20px;
        }
        
        .personality-radar {
          width: 300px;
          height: 300px;
        }
        
        .tuner-actions {
          display: flex;
          gap: 12px;
          justify-content: flex-end;
        }
        
        .reset-button,
        .save-button {
          padding: 8px 16px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .reset-button {
          background: rgba(255, 0, 102, 0.1);
          color: #ff0066;
          border-color: rgba(255, 0, 102, 0.2);
        }
        
        .reset-button:hover:not(:disabled) {
          background: rgba(255, 0, 102, 0.2);
          border-color: rgba(255, 0, 102, 0.3);
        }
        
        .save-button {
          background: rgba(0, 255, 136, 0.1);
          color: #00ff88;
          border-color: rgba(0, 255, 136, 0.2);
        }
        
        .save-button:hover:not(:disabled) {
          background: rgba(0, 255, 136, 0.2);
          border-color: rgba(0, 255, 136, 0.3);
        }
        
        .reset-button:disabled,
        .save-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default PersonalityTuner;