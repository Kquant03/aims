// src/ui/components/App.jsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ConsciousnessView from './consciousness/ConsciousnessView';
import MemoryExplorer from './memory/MemoryExplorer';
import ConversationInterface from './conversation/ConversationInterface';
import SystemControls from './controls/SystemControls';
import useWebSocket from './hooks/useWebSocket';
import useConsciousness from './hooks/useConsciousness';
import './App.css';

const App = () => {
  const [activeView, setActiveView] = useState('conversation');
  const [sessionId, setSessionId] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  
  // Custom hooks for state management
  const { messages, sendMessage, connectionState } = useWebSocket(sessionId);
  const { consciousnessState, updateConsciousness } = useConsciousness();
  
  // Initialize session on mount
  useEffect(() => {
    initializeSession();
  }, []);
  
  const initializeSession = async () => {
    try {
      const response = await fetch('/api/session', {
        method: 'GET',
        credentials: 'include'
      });
      const data = await response.json();
      setSessionId(data.session_id);
      setIsInitialized(true);
    } catch (error) {
      console.error('Failed to initialize session:', error);
    }
  };
  
  const navigationItems = [
    { id: 'conversation', label: 'Converse', icon: '💬' },
    { id: 'consciousness', label: 'Consciousness', icon: '🧠' },
    { id: 'memories', label: 'Memories', icon: '💎' },
    { id: 'controls', label: 'Controls', icon: '⚙️' }
  ];
  
  return (
    <div className="aims-app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">
            <span className="title-glow">AIMS</span>
            <span className="subtitle">Autonomous Intelligent Memory System</span>
          </h1>
          
          <div className="connection-status">
            <div className={`status-indicator ${connectionState}`} />
            <span>{connectionState === 'connected' ? 'Connected' : 'Connecting...'}</span>
          </div>
        </div>
      </header>
      
      {/* Navigation */}
      <nav className="main-navigation">
        {navigationItems.map(item => (
          <button
            key={item.id}
            className={`nav-item ${activeView === item.id ? 'active' : ''}`}
            onClick={() => setActiveView(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </button>
        ))}
      </nav>
      
      {/* Main Content */}
      <main className="app-content">
        <AnimatePresence mode="wait">
          {!isInitialized ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="loading-container"
            >
              <div className="loading-pulse">
                <div className="pulse-core" />
                <div className="pulse-ring" />
              </div>
              <p>Initializing consciousness...</p>
            </motion.div>
          ) : (
            <motion.div
              key={activeView}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="view-container"
            >
              {activeView === 'conversation' && (
                <ConversationInterface
                  sessionId={sessionId}
                  consciousnessState={consciousnessState}
                  onStateUpdate={updateConsciousness}
                />
              )}
              
              {activeView === 'consciousness' && (
                <ConsciousnessView
                  sessionId={sessionId}
                  consciousnessState={consciousnessState}
                />
              )}
              
              {activeView === 'memories' && (
                <MemoryExplorer
                  sessionId={sessionId}
                />
              )}
              
              {activeView === 'controls' && (
                <SystemControls
                  sessionId={sessionId}
                  onStateChange={updateConsciousness}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
      
      {/* Persistent Elements */}
      <div className="persistent-ui">
        {/* Coherence indicator */}
        <div className="coherence-indicator">
          <div 
            className="coherence-bar"
            style={{ 
              width: `${(consciousnessState?.coherence || 0) * 100}%`,
              backgroundColor: getCoherenceColor(consciousnessState?.coherence)
            }}
          />
        </div>
      </div>
    </div>
  );
};

const getCoherenceColor = (coherence) => {
  if (!coherence) return '#666';
  if (coherence > 0.8) return '#00ff88';
  if (coherence > 0.6) return '#ffaa00';
  if (coherence > 0.4) return '#ff6600';
  return '#ff0066';
};

export default App;