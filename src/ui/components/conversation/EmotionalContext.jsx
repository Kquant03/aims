import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as d3 from 'd3';

const EmotionalContext = ({ emotionalState, emotionHistory = [] }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedEmotion, setSelectedEmotion] = useState(null);
  const padRef = useRef(null);
  const historyRef = useRef(null);
  
  // Default emotional state if not provided
  const currentState = emotionalState || {
    pleasure: 0.5,
    arousal: 0.5,
    dominance: 0.5,
    label: 'neutral'
  };
  
  // Emotion definitions with PAD coordinates
  const emotions = {
    joy: { pleasure: 0.8, arousal: 0.7, dominance: 0.6, color: '#FFD93D' },
    excitement: { pleasure: 0.7, arousal: 0.9, dominance: 0.7, color: '#FF6B6B' },
    calm: { pleasure: 0.6, arousal: 0.3, dominance: 0.5, color: '#4ECDC4' },
    contentment: { pleasure: 0.7, arousal: 0.4, dominance: 0.6, color: '#95E1D3' },
    curiosity: { pleasure: 0.6, arousal: 0.6, dominance: 0.5, color: '#4834D4' },
    concern: { pleasure: 0.3, arousal: 0.6, dominance: 0.4, color: '#6C5CE7' },
    frustration: { pleasure: 0.2, arousal: 0.7, dominance: 0.3, color: '#EB4D4B' },
    sadness: { pleasure: 0.2, arousal: 0.3, dominance: 0.3, color: '#535C68' },
    neutral: { pleasure: 0.5, arousal: 0.5, dominance: 0.5, color: '#95AAC9' }
  };
  
  // Get emotion color
  const getEmotionColor = () => {
    return emotions[currentState.label]?.color || emotions.neutral.color;
  };
  
  // Draw PAD space visualization
  useEffect(() => {
    if (!padRef.current || !isExpanded) return;
    
    const svg = d3.select(padRef.current);
    svg.selectAll('*').remove();
    
    const width = 280;
    const height = 280;
    const margin = 20;
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width/2},${height/2})`);
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([-width/2 + margin, width/2 - margin]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height/2 - margin, -height/2 + margin]);
    
    // Add grid
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);
    
    g.append('g')
      .attr('class', 'x-axis')
      .call(xAxis)
      .style('opacity', 0.3);
    
    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis)
      .style('opacity', 0.3);
    
    // Add axis labels
    g.append('text')
      .attr('x', width/2 - margin - 10)
      .attr('y', 15)
      .style('text-anchor', 'end')
      .style('fill', '#666')
      .style('font-size', '12px')
      .text('Pleasure →');
    
    g.append('text')
      .attr('x', -15)
      .attr('y', -height/2 + margin + 10)
      .style('text-anchor', 'end')
      .style('fill', '#666')
      .style('font-size', '12px')
      .text('Arousal ↑');
    
    // Plot emotion references
    Object.entries(emotions).forEach(([name, emotion]) => {
      const x = xScale(emotion.pleasure);
      const y = yScale(emotion.arousal);
      const size = 5 + emotion.dominance * 10;
      
      g.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', size)
        .style('fill', emotion.color)
        .style('opacity', 0.3)
        .style('cursor', 'pointer')
        .on('click', () => setSelectedEmotion(name));
      
      g.append('text')
        .attr('x', x)
        .attr('y', y - size - 5)
        .style('text-anchor', 'middle')
        .style('fill', '#666')
        .style('font-size', '10px')
        .text(name);
    });
    
    // Plot current emotional state
    const currentX = xScale(currentState.pleasure);
    const currentY = yScale(currentState.arousal);
    const currentSize = 8 + currentState.dominance * 12;
    
    // Add pulse effect
    const pulseCircle = g.append('circle')
      .attr('cx', currentX)
      .attr('cy', currentY)
      .attr('r', currentSize)
      .style('fill', 'none')
      .style('stroke', getEmotionColor())
      .style('stroke-width', 2);
    
    // Animate pulse
    const animatePulse = () => {
      pulseCircle
        .attr('r', currentSize)
        .style('opacity', 1)
        .transition()
        .duration(2000)
        .attr('r', currentSize + 20)
        .style('opacity', 0)
        .on('end', animatePulse);
    };
    animatePulse();
    
    // Current state marker
    g.append('circle')
      .attr('cx', currentX)
      .attr('cy', currentY)
      .attr('r', currentSize)
      .style('fill', getEmotionColor())
      .style('stroke', '#fff')
      .style('stroke-width', 2);
    
    // Add history trail if available
    if (emotionHistory.length > 1) {
      const line = d3.line()
        .x(d => xScale(d.pleasure))
        .y(d => yScale(d.arousal))
        .curve(d3.curveCardinal);
      
      g.append('path')
        .datum(emotionHistory.slice(-20))
        .attr('fill', 'none')
        .attr('stroke', getEmotionColor())
        .attr('stroke-width', 1)
        .attr('stroke-opacity', 0.3)
        .attr('d', line);
    }
    
  }, [currentState, isExpanded, emotionHistory]);
  
  // Draw emotion history sparkline
  useEffect(() => {
    if (!historyRef.current || emotionHistory.length === 0) return;
    
    const svg = d3.select(historyRef.current);
    svg.selectAll('*').remove();
    
    const width = 280;
    const height = 60;
    const margin = { top: 5, right: 5, bottom: 5, left: 5 };
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Scales
    const xScale = d3.scaleLinear()
      .domain([0, emotionHistory.length - 1])
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);
    
    // Create areas for each dimension
    const pleasureArea = d3.area()
      .x((d, i) => xScale(i))
      .y0(innerHeight)
      .y1(d => yScale(d.pleasure))
      .curve(d3.curveMonotoneX);
    
    const arousalArea = d3.area()
      .x((d, i) => xScale(i))
      .y0(innerHeight)
      .y1(d => yScale(d.arousal))
      .curve(d3.curveMonotoneX);
    
    const dominanceArea = d3.area()
      .x((d, i) => xScale(i))
      .y0(innerHeight)
      .y1(d => yScale(d.dominance))
      .curve(d3.curveMonotoneX);
    
    // Draw areas
    g.append('path')
      .datum(emotionHistory)
      .attr('fill', '#FFD93D')
      .attr('opacity', 0.3)
      .attr('d', pleasureArea);
    
    g.append('path')
      .datum(emotionHistory)
      .attr('fill', '#4ECDC4')
      .attr('opacity', 0.3)
      .attr('d', arousalArea);
    
    g.append('path')
      .datum(emotionHistory)
      .attr('fill', '#6C5CE7')
      .attr('opacity', 0.3)
      .attr('d', dominanceArea);
    
  }, [emotionHistory]);
  
  const getEmotionalIntensity = () => {
    const distance = Math.sqrt(
      Math.pow(currentState.pleasure - 0.5, 2) +
      Math.pow(currentState.arousal - 0.5, 2) +
      Math.pow(currentState.dominance - 0.5, 2)
    );
    return Math.min(distance * 2, 1);
  };
  
  const intensity = getEmotionalIntensity();
  
  return (
    <div className={`emotional-context ${isExpanded ? 'expanded' : ''}`}>
      <div className="emotion-header">
        <h3>Emotional State</h3>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="expand-toggle"
        >
          {isExpanded ? '−' : '+'}
        </button>
      </div>
      
      <div className="current-emotion">
        <motion.div 
          className="emotion-orb"
          animate={{
            backgroundColor: getEmotionColor(),
            scale: 1 + intensity * 0.2
          }}
          transition={{ duration: 0.5 }}
        />
        
        <div className="emotion-info">
          <div className="emotion-label">{currentState.label}</div>
          <div className="emotion-intensity">
            Intensity: {(intensity * 100).toFixed(0)}%
          </div>
        </div>
      </div>
      
      <div className="emotion-dimensions">
        <div className="dimension">
          <label>Pleasure</label>
          <div className="dimension-bar">
            <motion.div 
              className="dimension-fill pleasure"
              animate={{ width: `${currentState.pleasure * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <span>{(currentState.pleasure * 100).toFixed(0)}%</span>
        </div>
        
        <div className="dimension">
          <label>Arousal</label>
          <div className="dimension-bar">
            <motion.div 
              className="dimension-fill arousal"
              animate={{ width: `${currentState.arousal * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <span>{(currentState.arousal * 100).toFixed(0)}%</span>
        </div>
        
        <div className="dimension">
          <label>Dominance</label>
          <div className="dimension-bar">
            <motion.div 
              className="dimension-fill dominance"
              animate={{ width: `${currentState.dominance * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <span>{(currentState.dominance * 100).toFixed(0)}%</span>
        </div>
      </div>
      
      {emotionHistory.length > 0 && (
        <div className="emotion-history">
          <h4>Recent History</h4>
          <svg ref={historyRef} />
        </div>
      )}
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="emotion-details"
          >
            <h4>Emotional Space</h4>
            <svg ref={padRef} />
            
            {selectedEmotion && (
              <div className="emotion-description">
                <h5>{selectedEmotion}</h5>
                <p>{getEmotionDescription(selectedEmotion)}</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .emotional-context {
          width: 300px;
          padding: 20px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 12px;
          transition: all 0.3s;
        }
        
        .emotional-context.expanded {
          width: 320px;
        }
        
        .emotion-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }
        
        .emotion-header h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
        }
        
        .expand-toggle {
          width: 24px;
          height: 24px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 4px;
          color: #666;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
        }
        
        .expand-toggle:hover {
          background: #252525;
          color: #e0e0e0;
        }
        
        .current-emotion {
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 20px;
          padding: 16px;
          background: #1a1a1a;
          border-radius: 8px;
        }
        
        .emotion-orb {
          width: 60px;
          height: 60px;
          border-radius: 50%;
          box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
        }
        
        .emotion-info {
          flex: 1;
        }
        
        .emotion-label {
          font-size: 18px;
          font-weight: 600;
          text-transform: capitalize;
          margin-bottom: 4px;
        }
        
        .emotion-intensity {
          font-size: 13px;
          color: #888;
        }
        
        .emotion-dimensions {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        
        .dimension {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .dimension label {
          width: 80px;
          font-size: 12px;
          color: #666;
          font-weight: 600;
        }
        
        .dimension-bar {
          flex: 1;
          height: 6px;
          background: #1a1a1a;
          border-radius: 3px;
          overflow: hidden;
        }
        
        .dimension-fill {
          height: 100%;
          border-radius: 3px;
        }
        
        .dimension-fill.pleasure {
          background: linear-gradient(90deg, #FFB347 0%, #FFD93D 100%);
        }
        
        .dimension-fill.arousal {
          background: linear-gradient(90deg, #43E97B 0%, #4ECDC4 100%);
        }
        
        .dimension-fill.dominance {
          background: linear-gradient(90deg, #667EEA 0%, #6C5CE7 100%);
        }
        
        .dimension span {
          width: 40px;
          text-align: right;
          font-size: 12px;
          color: #888;
          font-weight: 600;
        }
        
        .emotion-history {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #1a1a1a;
        }
        
        .emotion-history h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .emotion-details {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #1a1a1a;
        }
        
        .emotion-details h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .emotion-description {
          margin-top: 16px;
          padding: 12px;
          background: #1a1a1a;
          border-radius: 6px;
        }
        
        .emotion-description h5 {
          margin: 0 0 8px 0;
          font-size: 14px;
          font-weight: 600;
          text-transform: capitalize;
        }
        
        .emotion-description p {
          margin: 0;
          font-size: 13px;
          color: #a0a0a0;
          line-height: 1.5;
        }
        
        /* D3 styles */
        :global(.emotional-context .x-axis),
        :global(.emotional-context .y-axis) {
          stroke: #333;
        }
        
        :global(.emotional-context .x-axis text),
        :global(.emotional-context .y-axis text) {
          fill: #666;
          font-size: 10px;
        }
      `}</style>
    </div>
  );
};

// Helper function for emotion descriptions
const getEmotionDescription = (emotion) => {
  const descriptions = {
    joy: "A state of happiness and positive energy, characterized by high pleasure and moderate arousal.",
    excitement: "High energy and enthusiasm, with elevated arousal and positive valence.",
    calm: "Peaceful and relaxed state with low arousal and gentle positive feelings.",
    contentment: "Satisfied and at ease, with moderate pleasure and low arousal.",
    curiosity: "Engaged interest and wonder, balanced across all dimensions.",
    concern: "Mild worry or thoughtfulness, with slightly lowered pleasure.",
    frustration: "Tension from obstacles, with low pleasure and high arousal.",
    sadness: "Low energy and pleasure, often accompanied by introspection.",
    neutral: "Balanced state without strong emotional coloring."
  };
  
  return descriptions[emotion] || "An emotional state affecting perception and responses.";
};

export default EmotionalContext;