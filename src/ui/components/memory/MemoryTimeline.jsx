import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';

const MemoryTimeline = ({ memories, onMemorySelect, selectedId }) => {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });
  const [timeRange, setTimeRange] = useState('all');
  const [hoveredMemory, setHoveredMemory] = useState(null);
  const [brushSelection, setBrushSelection] = useState(null);
  
  // Handle container resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Filter memories by time range
  const getFilteredMemories = () => {
    if (!memories || memories.length === 0) return [];
    
    const now = new Date();
    const ranges = {
      day: 24 * 60 * 60 * 1000,
      week: 7 * 24 * 60 * 60 * 1000,
      month: 30 * 24 * 60 * 60 * 1000,
      year: 365 * 24 * 60 * 60 * 1000
    };
    
    if (timeRange === 'all') return memories;
    
    const cutoff = new Date(now.getTime() - ranges[timeRange]);
    return memories.filter(m => new Date(m.timestamp) >= cutoff);
  };
  
  // Draw timeline
  useEffect(() => {
    if (!svgRef.current) return;
    
    const filteredMemories = getFilteredMemories();
    if (filteredMemories.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const { width, height } = dimensions;
    const margin = { top: 40, right: 40, bottom: 100, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Sort memories by timestamp
    const sortedMemories = [...filteredMemories].sort((a, b) => 
      new Date(a.timestamp) - new Date(b.timestamp)
    );
    
    // Create scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(sortedMemories, d => new Date(d.timestamp)))
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);
    
    // Create color scale for memory types
    const colorScale = d3.scaleOrdinal()
      .domain(['conversation', 'insight', 'emotion', 'goal', 'learning'])
      .range(['#00a8ff', '#ffaa00', '#ff00aa', '#00ff88', '#aa00ff']);
    
    // Add axes
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickFormat(d3.timeFormat('%b %d'))
        .ticks(d3.timeDay.every(1)));
    
    xAxis.selectAll('text')
      .style('fill', '#666')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');
    
    const yAxis = g.append('g')
      .call(d3.axisLeft(yScale)
        .tickFormat(d => `${d * 100}%`)
        .ticks(5));
    
    yAxis.selectAll('text')
      .style('fill', '#666');
    
    // Add axis labels
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (innerHeight / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .style('fill', '#888')
      .style('font-size', '12px')
      .text('Importance');
    
    // Add grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickSize(-innerHeight)
        .tickFormat(''))
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);
    
    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale)
        .tickSize(-innerWidth)
        .tickFormat(''))
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);
    
    // Create timeline area
    const area = d3.area()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(innerHeight)
      .y1(d => yScale(d.importance))
      .curve(d3.curveMonotoneX);
    
    // Add area fill
    g.append('path')
      .datum(sortedMemories)
      .attr('fill', 'url(#timeline-gradient)')
      .attr('opacity', 0.3)
      .attr('d', area);
    
    // Add line
    const line = d3.line()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.importance))
      .curve(d3.curveMonotoneX);
    
    g.append('path')
      .datum(sortedMemories)
      .attr('fill', 'none')
      .attr('stroke', '#00a8ff')
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Add gradient definition
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'timeline-gradient')
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', 0).attr('y1', yScale(0))
      .attr('x2', 0).attr('y2', yScale(1));
    
    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#00a8ff')
      .attr('stop-opacity', 0.1);
    
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#00a8ff')
      .attr('stop-opacity', 0.5);
    
    // Add memory points
    const memories = g.selectAll('.memory-point')
      .data(sortedMemories)
      .enter().append('g')
      .attr('class', 'memory-point')
      .attr('transform', d => `translate(${xScale(new Date(d.timestamp))},${yScale(d.importance)})`);
    
    // Add circles
    memories.append('circle')
      .attr('r', d => 4 + d.importance * 6)
      .attr('fill', d => colorScale(d.type || 'conversation'))
      .attr('stroke', d => d.id === selectedId ? '#fff' : 'none')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        onMemorySelect(d);
      })
      .on('mouseover', (event, d) => {
        setHoveredMemory(d);
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', d => 6 + d.importance * 8);
      })
      .on('mouseout', (event, d) => {
        setHoveredMemory(null);
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', d => 4 + d.importance * 6);
      });
    
    // Add emotional state indicators
    memories.filter(d => d.emotional_context)
      .append('circle')
      .attr('r', 12)
      .attr('fill', 'none')
      .attr('stroke', d => {
        const emotion = d.emotional_context.label;
        const emotionColors = {
          joy: '#FFD93D',
          excitement: '#FF6B6B',
          calm: '#4ECDC4',
          curiosity: '#4834D4',
          concern: '#6C5CE7',
          neutral: '#95AAC9'
        };
        return emotionColors[emotion] || '#95AAC9';
      })
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '2,2')
      .attr('opacity', 0.6);
    
    // Add brush for selection
    const brush = d3.brushX()
      .extent([[0, 0], [innerWidth, innerHeight]])
      .on('end', brushended);
    
    const brushGroup = g.append('g')
      .attr('class', 'brush')
      .call(brush);
    
    function brushended(event) {
      if (!event.selection) {
        setBrushSelection(null);
        return;
      }
      
      const [x0, x1] = event.selection;
      const selectedMemories = sortedMemories.filter(d => {
        const x = xScale(new Date(d.timestamp));
        return x >= x0 && x <= x1;
      });
      
      setBrushSelection({
        start: xScale.invert(x0),
        end: xScale.invert(x1),
        memories: selectedMemories
      });
    }
    
    // Style brush
    brushGroup.selectAll('.selection')
      .style('fill', '#00a8ff')
      .style('fill-opacity', 0.1)
      .style('stroke', '#00a8ff')
      .style('stroke-width', 1);
    
  }, [memories, dimensions, timeRange, selectedId, onMemorySelect]);
  
  const timeRanges = [
    { value: 'all', label: 'All Time' },
    { value: 'day', label: 'Today' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' },
    { value: 'year', label: 'This Year' }
  ];
  
  return (
    <div className="memory-timeline" ref={containerRef}>
      <div className="timeline-header">
        <h3>Memory Timeline</h3>
        <div className="time-range-selector">
          {timeRanges.map(range => (
            <button
              key={range.value}
              className={`range-button ${timeRange === range.value ? 'active' : ''}`}
              onClick={() => setTimeRange(range.value)}
            >
              {range.label}
            </button>
          ))}
        </div>
      </div>
      
      <svg 
        ref={svgRef} 
        width={dimensions.width} 
        height={dimensions.height}
        className="timeline-svg"
      />
      
      {brushSelection && (
        <motion.div 
          className="selection-info"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h4>Selected Range</h4>
          <p>
            {brushSelection.start.toLocaleDateString()} - {brushSelection.end.toLocaleDateString()}
          </p>
          <p>{brushSelection.memories.length} memories selected</p>
          <button 
            onClick={() => setBrushSelection(null)}
            className="clear-selection"
          >
            Clear Selection
          </button>
        </motion.div>
      )}
      
      <AnimatePresence>
        {hoveredMemory && (
          <motion.div 
            className="timeline-tooltip"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="tooltip-date">
              {new Date(hoveredMemory.timestamp).toLocaleString()}
            </div>
            <div className="tooltip-type" style={{ color: getTypeColor(hoveredMemory.type) }}>
              {hoveredMemory.type || 'memory'}
            </div>
            <p className="tooltip-content">{hoveredMemory.content}</p>
            <div className="tooltip-footer">
              <span>Importance: {(hoveredMemory.importance * 100).toFixed(0)}%</span>
              {hoveredMemory.emotional_context && (
                <span className="emotion-tag">
                  {hoveredMemory.emotional_context.label}
                </span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .memory-timeline {
          position: relative;
          width: 100%;
          background: #0a0a0a;
          border-radius: 12px;
          border: 1px solid #333;
          padding: 20px;
        }
        
        .timeline-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        
        .timeline-header h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }
        
        .time-range-selector {
          display: flex;
          gap: 4px;
          background: #1a1a1a;
          padding: 4px;
          border-radius: 6px;
        }
        
        .range-button {
          padding: 6px 12px;
          background: none;
          border: none;
          color: #666;
          font-size: 12px;
          cursor: pointer;
          border-radius: 4px;
          transition: all 0.2s;
        }
        
        .range-button:hover {
          color: #e0e0e0;
        }
        
        .range-button.active {
          background: #333;
          color: #e0e0e0;
        }
        
        .timeline-svg {
          display: block;
        }
        
        .selection-info {
          position: absolute;
          top: 20px;
          left: 20px;
          padding: 16px;
          background: rgba(26, 26, 26, 0.95);
          border: 1px solid #333;
          border-radius: 8px;
          backdrop-filter: blur(10px);
        }
        
        .selection-info h4 {
          margin: 0 0 8px 0;
          font-size: 14px;
          font-weight: 600;
        }
        
        .selection-info p {
          margin: 0 0 4px 0;
          font-size: 13px;
          color: #888;
        }
        
        .clear-selection {
          margin-top: 8px;
          padding: 6px 12px;
          background: #252525;
          border: 1px solid #333;
          border-radius: 4px;
          color: #a0a0a0;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .clear-selection:hover {
          background: #333;
          color: #e0e0e0;
        }
        
        .timeline-tooltip {
          position: fixed;
          max-width: 300px;
          padding: 12px;
          background: rgba(26, 26, 26, 0.95);
          border: 1px solid #333;
          border-radius: 8px;
          backdrop-filter: blur(10px);
          pointer-events: none;
          z-index: 1000;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
        }
        
        .tooltip-date {
          font-size: 12px;
          color: #666;
          margin-bottom: 4px;
        }
        
        .tooltip-type {
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 8px;
        }
        
        .tooltip-content {
          margin: 0 0 8px 0;
          font-size: 13px;
          color: #e0e0e0;
          line-height: 1.5;
        }
        
        .tooltip-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 11px;
          color: #888;
        }
        
        .emotion-tag {
          padding: 2px 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
          text-transform: capitalize;
        }
        
        /* D3 styles */
        :global(.memory-timeline .grid line) {
          stroke: #333;
        }
        
        :global(.memory-timeline .grid path) {
          stroke-width: 0;
        }
        
        :global(.memory-timeline .brush .selection) {
          rx: 4;
        }
      `}</style>
    </div>
  );
};

// Helper function for type colors
const getTypeColor = (type) => {
  const colors = {
    conversation: '#00a8ff',
    insight: '#ffaa00',
    emotion: '#ff00aa',
    goal: '#00ff88',
    learning: '#aa00ff',
    milestone: '#ff0066'
  };
  return colors[type] || '#666';
};

export default MemoryTimeline;