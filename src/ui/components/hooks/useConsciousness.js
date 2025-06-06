// src/ui/components/hooks/useConsciousness.js
import { useState, useEffect, useCallback } from 'react';

const useConsciousness = () => {
  const [consciousnessState, setConsciousnessState] = useState({
    coherence: 0.7,
    attention_focus: null,
    emotional_state: {
      pleasure: 0.5,
      arousal: 0.5,
      dominance: 0.5,
      label: 'neutral'
    },
    personality: {
      openness: 0.8,
      conscientiousness: 0.7,
      extraversion: 0.6,
      agreeableness: 0.8,
      neuroticism: 0.3
    },
    working_memory: [],
    working_memory_size: 0,
    interaction_count: 0
  });
  
  // Fetch initial state
  useEffect(() => {
    fetchConsciousnessState();
    
    // Listen for WebSocket updates
    const handleUpdate = (event) => {
      const update = event.detail;
      setConsciousnessState(prev => ({
        ...prev,
        ...update
      }));
    };
    
    window.addEventListener('consciousness_update', handleUpdate);
    
    return () => {
      window.removeEventListener('consciousness_update', handleUpdate);
    };
  }, []);
  
  const fetchConsciousnessState = async () => {
    try {
      const response = await fetch('/api/consciousness/state', {
        credentials: 'include'
      });
      const data = await response.json();
      setConsciousnessState(data);
    } catch (error) {
      console.error('Failed to fetch consciousness state:', error);
    }
  };
  
  const updateConsciousness = useCallback((updates) => {
    setConsciousnessState(prev => ({
      ...prev,
      ...updates
    }));
  }, []);
  
  return { consciousnessState, updateConsciousness };
};

export default useConsciousness;