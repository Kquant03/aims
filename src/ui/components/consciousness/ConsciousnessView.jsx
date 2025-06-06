// src/ui/components/consciousness/ConsciousnessView.jsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import CoreVisualization from './CoreVisualization';
import EmotionalLandscape from './EmotionalLandscape';
import AttentionFocus from './AttentionFocus';
import CoherenceMetrics from './CoherenceMetrics';
import './ConsciousnessView.css';

const ConsciousnessView = ({ sessionId, consciousnessState }) => {
  const [selectedView, setSelectedView] = useState('overview');
  const [historicalData, setHistoricalData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    fetchConsciousnessHistory();
  }, [sessionId]);
  
  const fetchConsciousnessHistory = async () => {
    try {
      const response = await fetch(`/api/consciousness/history?session_id=${sessionId}`, {
        credentials: 'include'
      });
      const data = await response.json();
      setHistoricalData(data);
    } catch (error) {
      console.error('Failed to fetch consciousness history:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const views = [
    { id: 'overview', label: 'Overview', icon: '🎯' },
    { id: 'emotional', label: 'Emotional', icon: '💭' },
    { id: 'attention', label: 'Attention', icon: '👁️' },
    { id: 'metrics', label: 'Metrics', icon: '📊' }
  ];
  
  return (
    <div className="consciousness-view">
      <div className="view-header">
        <h2>Consciousness State</h2>
        
        <div className="view-tabs">
          {views.map(view => (
            <button
              key={view.id}
              className={`view-tab ${selectedView === view.id ? 'active' : ''}`}
              onClick={() => setSelectedView(view.id)}
            >
              <span className="tab-icon">{view.icon}</span>
              <span className="tab-label">{view.label}</span>
            </button>
          ))}
        </div>
      </div>
      
      <div className="view-content">
        {isLoading ? (
          <div className="loading-state">
            <div className="consciousness-loader">
              <div className="loader-brain" />
              <p>Loading consciousness data...</p>
            </div>
          </div>
        ) : (
          <motion.div
            key={selectedView}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="consciousness-panel"
          >
            {selectedView === 'overview' && (
              <div className="overview-grid">
                <div className="panel-section core-section">
                  <h3>Consciousness Core</h3>
                  <CoreVisualization
                    coherence={consciousnessState?.coherence || 0.7}
                    emotionalState={consciousnessState?.emotional_state || {}}
                    workingMemory={consciousnessState?.working_memory || []}
                  />
                </div>
                
                <div className="panel-section stats-section">
                  <h3>Current State</h3>
                  <div className="state-metrics">
                    <div className="metric">
                      <label>Coherence</label>
                      <div className="metric-value">
                        {((consciousnessState?.coherence || 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="metric">
                      <label>Working Memory</label>
                      <div className="metric-value">
                        {consciousnessState?.working_memory_size || 0} items
                      </div>
                    </div>
                    <div className="metric">
                      <label>Interactions</label>
                      <div className="metric-value">
                        {consciousnessState?.interaction_count || 0}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="panel-section personality-section">
                  <h3>Personality Traits</h3>
                  <PersonalityRadar 
                    traits={consciousnessState?.personality || {}}
                  />
                </div>
              </div>
            )}
            
            {selectedView === 'emotional' && (
              <EmotionalLandscape
                currentState={consciousnessState?.emotional_state}
                history={historicalData?.emotion_history || []}
              />
            )}
            
            {selectedView === 'attention' && (
              <AttentionFocus
                currentFocus={consciousnessState?.attention_focus}
                focusHistory={historicalData?.attention_history || []}
                dimensions={consciousnessState?.attention_dimensions}
              />
            )}
            
            {selectedView === 'metrics' && (
              <CoherenceMetrics
                currentCoherence={consciousnessState?.coherence}
                history={historicalData?.coherence_history || []}
                evolutionEvents={historicalData?.evolution_events || []}
              />
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
};

// Personality Radar Component
const PersonalityRadar = ({ traits }) => {
  const dimensions = [
    { key: 'openness', label: 'Openness' },
    { key: 'conscientiousness', label: 'Conscientiousness' },
    { key: 'extraversion', label: 'Extraversion' },
    { key: 'agreeableness', label: 'Agreeableness' },
    { key: 'neuroticism', label: 'Neuroticism' }
  ];
  
  return (
    <div className="personality-radar">
      {dimensions.map(dim => (
        <div key={dim.key} className="trait-bar">
          <label>{dim.label}</label>
          <div className="trait-progress">
            <div 
              className="trait-fill"
              style={{ 
                width: `${(traits[dim.key] || 0.5) * 100}%`,
                backgroundColor: getTraitColor(dim.key)
              }}
            />
          </div>
          <span className="trait-value">
            {((traits[dim.key] || 0.5) * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
};

const getTraitColor = (trait) => {
  const colors = {
    openness: '#00a8ff',
    conscientiousness: '#00ff88',
    extraversion: '#ffaa00',
    agreeableness: '#ff00ff',
    neuroticism: '#ff0066'
  };
  return colors[trait] || '#666';
};

export default ConsciousnessView;