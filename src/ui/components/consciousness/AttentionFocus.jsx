import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

const AttentionFocus = ({ currentFocus, focusHistory = [], dimensions }) => {
  const radarRef = useRef(null);
  const timelineRef = useRef(null);
  const [selectedDimension, setSelectedDimension] = useState(null);
  
  // Default attention dimensions if not provided
  const defaultDimensions = [
    { key: 'emotional', label: 'Emotional', color: '#ff00aa' },
    { key: 'analytical', label: 'Analytical', color: '#00a8ff' },
    { key: 'creative', label: 'Creative', color: '#00ff88' },
    { key: 'personal', label: 'Personal', color: '#ffaa00' },
    { key: 'philosophical', label: 'Philosophical', color: '#aa00ff' },
    { key: 'practical', label: 'Practical', color: '#00ffaa' }
  ];
  
  const activeDimensions = dimensions || defaultDimensions;
  
  // Parse current focus to get dimension scores
  const getDimensionScores = () => {
    if (!currentFocus) return {};
    
    const scores = {};
    activeDimensions.forEach(dim => {
      // Extract score from focus string or use default
      const regex = new RegExp(`${dim.key}\\s*:\\s*(\\d+(?:\\.\\d+)?)`);
      const match = currentFocus.match(regex);
      scores[dim.key] = match ? parseFloat(match[1]) / 100 : Math.random() * 0.5 + 0.3;
    });
    
    return scores;
  };
  
  const dimensionScores = getDimensionScores();
  
  // Draw radar chart
  useEffect(() => {
    if (!radarRef.current) return;
    
    const svg = d3.select(radarRef.current);
    svg.selectAll('*').remove();
    
    const width = 300;
    const height = 300;
    const margin = 40;
    const radius = Math.min(width, height) / 2 - margin;
    
    const angleSlice = (Math.PI * 2) / activeDimensions.length;
    
    // Create main group
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);
    
    // Create scales
    const rScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, radius]);
    
    // Draw grid circles
    const levels = 5;
    for (let level = 0; level < levels; level++) {
      const levelFactor = radius * ((level + 1) / levels);
      
      g.append('circle')
        .attr('r', levelFactor)
        .style('fill', 'none')
        .style('stroke', '#333')
        .style('stroke-width', '0.5px');
    }
    
    // Draw axis lines and labels
    const axis = g.selectAll('.axis')
      .data(activeDimensions)
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
    
    axis.append('text')
      .attr('class', 'legend')
      .style('font-size', '12px')
      .style('fill', '#888')
      .attr('text-anchor', 'middle')
      .attr('x', (d, i) => (rScale(1) + 20) * Math.cos(angleSlice * i - Math.PI / 2))
      .attr('y', (d, i) => (rScale(1) + 20) * Math.sin(angleSlice * i - Math.PI / 2))
      .text(d => d.label)
      .style('cursor', 'pointer')
      .on('click', (event, d) => setSelectedDimension(d));
    
    // Prepare data for radar area
    const radarData = activeDimensions.map((dim, i) => ({
      dimension: dim,
      value: dimensionScores[dim.key] || 0,
      angle: angleSlice * i
    }));
    
    // Create line generator
    const radarLine = d3.lineRadial()
      .radius(d => rScale(d.value))
      .angle(d => d.angle)
      .curve(d3.curveLinearClosed);
    
    // Draw the radar area
    g.append('path')
      .datum(radarData)
      .attr('d', radarLine)
      .style('fill', '#00a8ff')
      .style('fill-opacity', 0.2)
      .style('stroke', '#00a8ff')
      .style('stroke-width', 2);
    
    // Draw dots on vertices
    g.selectAll('.radar-dot')
      .data(radarData)
      .enter()
      .append('circle')
      .attr('class', 'radar-dot')
      .attr('r', 4)
      .attr('cx', d => rScale(d.value) * Math.cos(d.angle - Math.PI / 2))
      .attr('cy', d => rScale(d.value) * Math.sin(d.angle - Math.PI / 2))
      .style('fill', d => d.dimension.color)
      .style('stroke', '#fff')
      .style('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).transition().duration(200).attr('r', 6);
        
        // Show tooltip
        const tooltip = g.append('g').attr('class', 'tooltip');
        
        const rect = tooltip.append('rect')
          .attr('x', -30)
          .attr('y', -25)
          .attr('width', 60)
          .attr('height', 20)
          .attr('fill', '#1a1a1a')
          .attr('stroke', '#333')
          .attr('rx', 4);
        
        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -10)
          .style('fill', '#e0e0e0')
          .style('font-size', '12px')
          .text(`${(d.value * 100).toFixed(0)}%`);
        
        tooltip.attr('transform', 
          `translate(${rScale(d.value) * Math.cos(d.angle - Math.PI / 2)},
                     ${rScale(d.value) * Math.sin(d.angle - Math.PI / 2)})`
        );
      })
      .on('mouseout', function() {
        d3.select(this).transition().duration(200).attr('r', 4);
        g.select('.tooltip').remove();
      });
    
  }, [currentFocus, activeDimensions]);
  
  // Draw timeline
  useEffect(() => {
    if (!timelineRef.current || focusHistory.length === 0) return;
    
    const svg = d3.select(timelineRef.current);
    svg.selectAll('*').remove();
    
    const width = 600;
    const height = 150;
    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Process history data
    const processedData = focusHistory.map((item, i) => ({
      ...item,
      timestamp: new Date(item.timestamp || Date.now() - (focusHistory.length - i) * 60000)
    }));
    
    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(processedData, d => d.timestamp))
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);
    
    // Create area generator for each dimension
    activeDimensions.forEach(dim => {
      const area = d3.area()
        .x(d => xScale(d.timestamp))
        .y0(innerHeight)
        .y1(d => yScale(d.scores?.[dim.key] || 0))
        .curve(d3.curveMonotoneX);
      
      g.append('path')
        .datum(processedData)
        .attr('fill', dim.color)
        .attr('fill-opacity', 0.3)
        .attr('d', area);
    });
    
    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')));
    
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${d * 100}%`));
    
  }, [focusHistory, activeDimensions]);
  
  return (
    <div className="attention-focus">
      <div className="focus-header">
        <h3>Attention Distribution</h3>
        {currentFocus && (
          <div className="current-focus-text">
            "{currentFocus}"
          </div>
        )}
      </div>
      
      <div className="focus-content">
        <div className="radar-section">
          <svg ref={radarRef} />
          
          <div className="dimension-legend">
            {activeDimensions.map(dim => (
              <motion.div
                key={dim.key}
                className={`legend-item ${selectedDimension?.key === dim.key ? 'selected' : ''}`}
                onClick={() => setSelectedDimension(dim)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div 
                  className="legend-color"
                  style={{ backgroundColor: dim.color }}
                />
                <span className="legend-label">{dim.label}</span>
                <span className="legend-value">
                  {((dimensionScores[dim.key] || 0) * 100).toFixed(0)}%
                </span>
              </motion.div>
            ))}
          </div>
        </div>
        
        {focusHistory.length > 0 && (
          <div className="timeline-section">
            <h4>Focus Evolution</h4>
            <svg ref={timelineRef} />
          </div>
        )}
        
        {selectedDimension && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="dimension-detail"
          >
            <h4>{selectedDimension.label} Focus</h4>
            <p>
              Current attention on {selectedDimension.label.toLowerCase()} aspects: 
              <strong> {((dimensionScores[selectedDimension.key] || 0) * 100).toFixed(0)}%</strong>
            </p>
            <div className="focus-description">
              {getDimensionDescription(selectedDimension.key)}
            </div>
          </motion.div>
        )}
      </div>
      
      <style jsx>{`
        .attention-focus {
          padding: 20px;
          background: #0a0a0a;
          border-radius: 12px;
          border: 1px solid #333;
        }
        
        .focus-header {
          margin-bottom: 24px;
        }
        
        .focus-header h3 {
          margin: 0 0 12px 0;
          font-size: 18px;
          font-weight: 600;
        }
        
        .current-focus-text {
          font-size: 14px;
          color: #888;
          font-style: italic;
          padding: 12px;
          background: #1a1a1a;
          border-radius: 8px;
          border-left: 3px solid #00a8ff;
        }
        
        .focus-content {
          display: grid;
          gap: 24px;
        }
        
        .radar-section {
          display: flex;
          gap: 24px;
          align-items: center;
        }
        
        .dimension-legend {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          background: #1a1a1a;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s;
          border: 1px solid transparent;
        }
        
        .legend-item:hover {
          background: #252525;
          border-color: #333;
        }
        
        .legend-item.selected {
          border-color: #00a8ff;
          background: #252525;
        }
        
        .legend-color {
          width: 12px;
          height: 12px;
          border-radius: 50%;
        }
        
        .legend-label {
          flex: 1;
          font-size: 13px;
          color: #e0e0e0;
        }
        
        .legend-value {
          font-size: 12px;
          color: #888;
          font-weight: 600;
        }
        
        .timeline-section {
          margin-top: 24px;
        }
        
        .timeline-section h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .dimension-detail {
          padding: 16px;
          background: #1a1a1a;
          border-radius: 8px;
          border: 1px solid #333;
        }
        
        .dimension-detail h4 {
          margin: 0 0 8px 0;
          font-size: 16px;
          font-weight: 600;
        }
        
        .dimension-detail p {
          margin: 0 0 12px 0;
          color: #a0a0a0;
        }
        
        .dimension-detail strong {
          color: #00a8ff;
        }
        
        .focus-description {
          font-size: 14px;
          line-height: 1.6;
          color: #888;
        }
        
        /* D3 styles */
        :global(.radar-dot) {
          transition: r 0.2s;
        }
        
        :global(.axis text) {
          fill: #888;
        }
        
        :global(.axis line) {
          stroke: #333;
        }
      `}</style>
    </div>
  );
};

// Helper function for dimension descriptions
const getDimensionDescription = (dimension) => {
  const descriptions = {
    emotional: "Focused on understanding and processing emotional aspects, empathy, and feelings.",
    analytical: "Engaged in logical reasoning, problem-solving, and systematic analysis.",
    creative: "Exploring imaginative solutions, artistic expression, and novel ideas.",
    personal: "Attending to individual experiences, relationships, and personal growth.",
    philosophical: "Contemplating deeper meanings, ethics, and fundamental questions.",
    practical: "Concentrating on actionable solutions and real-world applications."
  };
  
  return descriptions[dimension] || "Focused on this aspect of the conversation.";
};

export default AttentionFocus;