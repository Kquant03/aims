import { useState } from 'react';

const useConsciousness = () => {
  const [consciousnessState, setConsciousnessState] = useState({
    coherence: 0.7,
    emotionalState: { pleasure: 0.5, arousal: 0.5, dominance: 0.5 }
  });
  
  const updateConsciousness = (newState) => {
    setConsciousnessState(prev => ({ ...prev, ...newState }));
  };
  
  return { consciousnessState, updateConsciousness };
};

export default useConsciousness;
