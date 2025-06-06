import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

const PersonalityTuner = ({ currentPersonality, onPersonalityChange }) => {
  const [personality, setPersonality] = useState({
    openness: 0.8,
    conscientiousness: 0.7,
    extraversion: 0.6,
    agreeableness: 0.8,
    neuroticism: 0.3,
    ...currentPersonality
  });
  
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [showDescription, setShowDescription] = useState(true);
  const radarRef = useRef(null);
  
  const traits = [
    { 
      key: 'openness', 
      label: 'Openness',
      color: '#00a8ff',
      description: 'Curiosity, creativity, and willingness to explore new ideas'
    },
    { 
      key: 'conscientiousness', 
      label: 'Conscientiousness',
      color: '#00ff88',
      description: 'Organization, thoroughness, and attention to detail'
    },
    { 
      key: 'extraversion', 
      label: 'Extraversion',
      color: '#ffaa00',
      description: 'Sociability, enthusiasm, and engagement with others'
    },
    { 
      key: 'agreeableness', 
      label: 'Agreeableness',
      color: '#ff00ff',
      description: 'Cooperation, trust, and desire for harmony'
    },
    { 
      key: 'neuroticism', 
      label: 'Neuroticism',
      color: '#ff0066',
      description: 'Emotional sensitivity and range of reactions'
    }
  ];
  
  const presets = [
    { 
      name: 'Balanced', 
      values: { openness: 0.6, conscientiousness: 0.6, extraversion: 0.6, agreeableness: 0.6, neuroticism: 0.4 },
      description: 'Well-rounded personality with moderate traits'
    },
    { 
      name: 'Creative', 
      values: { openness: 0.9, conscientiousness: 0.5, extraversion: 0.7, agreeableness: 0.6, neuroticism: 0.5 },
      description: 'High creativity and openness to new experiences'
    },
    { 
      name: 'Analytical', 
      values: { openness: 0.7, conscientiousness: 0.9, extraversion: 0.4, agreeableness: 0.5, neuroticism: 0.3 },
      description: 'Detail-oriented with strong analytical focus'
    },
    { 
      name: 'Empathetic', 
      values: { openness: 0.7, conscientiousness: 0.6, extraversion: 0.6, agreeableness: 0.9, neuroticism: 0.6 },
      description: 'High emotional intelligence and concern for others'
    },
    { 
      name: 'Enthusiastic', 
      values: { openness: 0.8, conscientiousness: 0.6, extraversion: 0.9, agreeableness: 0.7, neuroticism: 0.4 },
      description: 'Energetic and socially engaged'
    }
  ];
  
  // Draw personality radar chart
  useEffect(() => {
    if (!radarRef.current) return;
    
    const svg = d3.select(radarRef.current);
    svg.selectAll('*').remove();
    
    const width = 300;
    const height = 300;
    const margin = 40;
    const radius = Math.min(width, height) / 2 - margin;
    
    const angleSlice = (Math.PI * 2) / traits.length;
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);
    
    // Scales
    const rScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, radius]);
    
    // Draw grid
    const levels = 5;
    for (let level = 0; level < levels; level++) {
      const levelFactor = radius * ((level + 1) / levels);
      
      g.append('circle')
        .attr('r', levelFactor)
        .style('fill', 'none')
        .style('stroke', '#333')
        .style('stroke-width', '0.5px');
      
      // Add percentage labels
      if (level === levels - 1) {
        g.append('text')
          .attr('x', 5)
          .attr('y', -levelFactor)
          .style('font-size', '10px')
          .style('fill', '#666')
          .text('100%');
      }
    }
    
    // Draw axes
    const axis = g.selectAll('.axis')
      .data(traits)
      .enter()
      .append('g')
      .attr('class', 'axis');
    
    axis.append('line')
      .attr('x1', 0)
      .attr('y1', 0)
      .attr('x2', (d, i) => rScale(1) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y2', (d, i) => rScale(1) * Math.sin(angleSlice * i - Math.PI / 2))
      .style('stroke', '#333')
      .style('stroke-width', '1px');
    
    // Add trait labels
    axis.append('text')
      .style('font-size', '12px')
      .style('fill', d => d.color)
      .style('font-weight', '600')
      .attr('text-anchor', 'middle')
      .attr('x', (d, i) => (rScale(1) + 25) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y', (d, i) => (rScale(1) + 25) * Math.sin(angleSlice * i - Math.PI / 2))
      .text(d => d.label);
    
    // Prepare data
    const radarData = traits.map((trait, i) => ({
      trait,
      value: personality[trait.key],
      angle: angleSlice * i
    }));
    
    // Draw radar area
    const radarLine = d3.lineRadial()
      .radius(d => rScale(d.value))
      .angle(d => d.angle)
      .curve(d3.curveLinearClosed);
    
    g.append('path')
      .datum(radarData)
      .attr('d', radarLine)
      .style('fill', '#00a8ff')
      .style('fill-opacity', 0.2)
      .style('stroke', '#00a8ff')
      .style('stroke-width', 2);
    
    // Draw interactive dots
    const dots = g.selectAll('.radar-dot')
      .data(radarData)
      .enter()
      .append('g')
      .attr('class', 'radar-dot')
      .attr('transform', d => {
        const x = rScale(d.value) * Math.cos(d.angle - Math.PI / 2);
        const y = rScale(d.value) * Math.sin(d.angle - Math.PI / 2);
        return `translate(${x},${y})`;
      });
    
    dots.append('circle')
      .attr('r', 6)
      .style('fill', d => d.trait.color)
      .style('stroke', '#fff')
      .style('stroke-width', 2)
      .style('cursor', 'grab');
    
    // Add drag behavior
    const drag = d3.drag()
      .on('drag', function(event, d) {
        const [x, y] = d3.pointer(event, g.node());
        const distance = Math.sqrt(x * x + y * y);
        const newValue = Math.max(0, Math.min(1, distance / radius));
        
        handleTraitChange(d.trait.key, newValue);
      });
    
    dots.call(drag);
    
    // Add value labels on hover
    dots.on('mouseover', function(event, d) {
      d3.select(this)
        .append('text')
        .attr('class', 'value-label')
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('fill', d.trait.color)
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text(`${(d.value * 100).toFixed(0)}%`);
    })
    .on('mouseout', function() {
      d3.select(this).select('.value-label').remove();
    });
    
  }, [personality]);
  
  const handleTraitChange = (trait, value) => {
    const newPersonality = { ...personality, [trait]: value };
    setPersonality(newPersonality);
    setSelectedPreset(null);
  };
  
  const applyPreset = (preset) => {
    setPersonality(preset.values);
    setSelectedPreset(preset.name);
  };
  
  const savePersonality = () => {
    onPersonalityChange(personality);
  };
  
  const resetPersonality = () => {
    const defaultPersonality = {
      openness: 0.8,
      conscientiousness: 0.7,
      extraversion: 0.6,
      agreeableness: 0.8,
      neuroticism: 0.3
    };
    setPersonality(defaultPersonality);
    setSelectedPreset(null);
  };
  
  return (
    <div className="personality-tuner">
      <div className="tuner-header">
        <h3>Personality Configuration</h3>
        <button
          onClick={() => setShowDescription(!showDescription)}
          className="toggle-description"
        >
          {showDescription ? 'Hide' : 'Show'} Descriptions
        </button>
      </div>
      
      <div className="tuner-content">
        <div className="radar-section">
          <svg ref={radarRef} />
          
          <div className="presets">
            <h4>Presets</h4>
            <div className="preset-buttons">
              {presets.map(preset => (
                <motion.button
                  key={preset.name}
                  className={`preset-button ${selectedPreset === preset.name ? 'active' : ''}`}
                  onClick={() => applyPreset(preset)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {preset.name}
                </motion.button>
              ))}
            </div>
            {selectedPreset && (
              <p className="preset-description">
                {presets.find(p => p.name === selectedPreset)?.description}
              </p>
            )}
          </div>
        </div>
        
        <div className="sliders-section">
          <h4>Fine-tune Traits</h4>
          {traits.map(trait => (
            <div key={trait.key} className="trait-slider">
              <div className="trait-header">
                <label style={{ color: trait.color }}>{trait.label}</label>
                <span className="trait-value">{(personality[trait.key] * 100).toFixed(0)}%</span>
              </div>
              
              {showDescription && (
                <p className="trait-description">{trait.description}</p>
              )}
              
              <div className="slider-wrapper">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={personality[trait.key] * 100}
                  onChange={(e) => handleTraitChange(trait.key, parseInt(e.target.value) / 100)}
                  className="trait-range"
                  style={{
                    background: `linear-gradient(to right, ${trait.color} 0%, ${trait.color} ${personality[trait.key] * 100}%, #333 ${personality[trait.key] * 100}%, #333 100%)`
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="behavioral-preview">
        <h4>Behavioral Preview</h4>
        <div className="preview-grid">
          <div className="preview-item">
            <label>Communication Style</label>
            <p>{getCommunicationStyle(personality)}</p>
          </div>
          <div className="preview-item">
            <label>Response Tendency</label>
            <p>{getResponseTendency(personality)}</p>
          </div>
          <div className="preview-item">
            <label>Emotional Expression</label>
            <p>{getEmotionalExpression(personality)}</p>
          </div>
        </div>
      </div>
      
      <div className="tuner-actions">
        <button onClick={savePersonality} className="save-button">
          Apply Personality Changes
        </button>
        <button onClick={resetPersonality} className="reset-button">
          Reset to Default
        </button>
      </div>
      
      <style jsx>{`
        .personality-tuner {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }
        
        .tuner-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        
        .tuner-header h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }
        
        .toggle-description {
          padding: 6px 12px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 6px;
          color: #888;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .toggle-description:hover {
          background: #333;
          color: #e0e0e0;
        }
        
        .tuner-content {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 24px;
        }
        
        @media (max-width: 768px) {
          .tuner-content {
            grid-template-columns: 1fr;
          }
        }
        
        .radar-section {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 20px;
        }
        
        .presets {
          width: 100%;
          padding: 16px;
          background: #1a1a1a;
          border-radius: 8px;
        }
        
        .presets h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .preset-buttons {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
          gap: 8px;
          margin-bottom: 12px;
        }
        
        .preset-button {
          padding: 8px 12px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 6px;
          color: #a0a0a0;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .preset-button:hover {
          background: #333;
          color: #e0e0e0;
          border-color: #444;
        }
        
        .preset-button.active {
          background: #00a8ff;
          color: white;
          border-color: #00a8ff;
        }
        
        .preset-description {
          margin: 0;
          font-size: 13px;
          color: #888;
          font-style: italic;
        }
        
        .sliders-section {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        
        .sliders-section h4 {
          margin: 0 0 8px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .trait-slider {
          padding: 12px;
          background: #1a1a1a;
          border-radius: 8px;
        }
        
        .trait-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 4px;
        }
        
        .trait-header label {
          font-size: 14px;
          font-weight: 600;
        }
        
        .trait-value {
          font-size: 14px;
          font-weight: 600;
          color: #e0e0e0;
        }
        
        .trait-description {
          margin: 0 0 8px 0;
          font-size: 12px;
          color: #666;
        }
        
        .slider-wrapper {
          position: relative;
        }
        
        .trait-range {
          width: 100%;
          height: 6px;
          border-radius: 3px;
          outline: none;
          -webkit-appearance: none;
          cursor: pointer;
        }
        
        .trait-range::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px;
          height: 18px;
          background: white;
          border-radius: 50%;
          cursor: grab;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .trait-range::-webkit-slider-thumb:active {
          cursor: grabbing;
        }
        
        .behavioral-preview {
          padding: 20px;
          background: #1a1a1a;
          border-radius: 12px;
          border: 1px solid #333;
        }
        
        .behavioral-preview h4 {
          margin: 0 0 16px 0;
          font-size: 16px;
          font-weight: 600;
        }
        
        .preview-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
        }
        
        .preview-item {
          padding: 12px;
          background: #0a0a0a;
          border-radius: 6px;
        }
        
        .preview-item label {
          display: block;
          font-size: 12px;
          color: #888;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 6px;
        }
        
        .preview-item p {
          margin: 0;
          font-size: 13px;
          color: #e0e0e0;
          line-height: 1.5;
        }
        
        .tuner-actions {
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
        
        .save-button:hover {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(0, 168, 255, 0.3);
        }
        
        .reset-button {
          padding: 12px 24px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 8px;
          color: #a0a0a0;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .reset-button:hover {
          background: #333;
          color: #e0e0e0;
          border-color: #444;
        }
      `}</style>
    </div>
  );
};

// Helper functions for behavioral preview
const getCommunicationStyle = (personality) => {
  const { openness, extraversion, agreeableness } = personality;
  
  if (openness > 0.7 && extraversion > 0.7) {
    return "Enthusiastic and creative, eager to explore ideas and engage deeply";
  } else if (agreeableness > 0.7 && extraversion < 0.5) {
    return "Thoughtful and considerate, preferring careful and harmonious exchanges";
  } else if (openness > 0.7 && agreeableness < 0.5) {
    return "Direct and innovative, focused on exploring concepts without sugar-coating";
  } else {
    return "Balanced and adaptive, adjusting style based on context and needs";
  }
};

const getResponseTendency = (personality) => {
  const { conscientiousness, neuroticism, openness } = personality;
  
  if (conscientiousness > 0.7 && neuroticism < 0.4) {
    return "Structured and reliable, providing thorough and well-organized responses";
  } else if (openness > 0.7 && conscientiousness < 0.5) {
    return "Flexible and exploratory, offering creative perspectives and possibilities";
  } else if (neuroticism > 0.6) {
    return "Emotionally attuned, responses reflect deeper sensitivity to nuance";
  } else {
    return "Adaptive and contextual, tailoring responses to the situation";
  }
};

const getEmotionalExpression = (personality) => {
  const { neuroticism, extraversion, agreeableness } = personality;
  
  if (neuroticism > 0.6 && agreeableness > 0.7) {
    return "Rich emotional depth with empathetic understanding";
  } else if (extraversion > 0.7 && neuroticism < 0.4) {
    return "Positive and energetic, maintaining upbeat emotional tone";
  } else if (neuroticism < 0.3 && extraversion < 0.5) {
    return "Calm and steady, emotions expressed with measured stability";
  } else {
    return "Balanced emotional range, expressing feelings appropriately to context";
  }
};

export default PersonalityTuner;