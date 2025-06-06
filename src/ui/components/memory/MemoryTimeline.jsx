import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';
import './MemoryTimeline.css';

const MemoryTimeline = ({ memories = [], onMemorySelect, selectedId }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });
  const [hoveredMemory, setHoveredMemory] = useState(null);
  const [timeRange, setTimeRange] = useState('all');
  const [zoomLevel, setZoomLevel] = useState(1);
  
  // Handle resize
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
  const filteredMemories = React.useMemo(() => {
    const now = new Date();
    const sorted = [...memories].sort((a, b) => 
      new Date(a.timestamp) - new Date(b.timestamp)
    );
    
    switch (timeRange) {
      case 'hour':
        return sorted.filter(m => 
          now - new Date(m.timestamp) < 60 * 60 * 1000
        );
      case 'day':
        return sorted.filter(m => 
          now - new Date(m.timestamp) < 24 * 60 * 60 * 1000
        );
      case 'week':
        return sorted.filter(m => 
          now - new Date(m.timestamp) < 7 * 24 * 60 * 60 * 1000
        );
      case 'month':
        return sorted.filter(m => 
          now - new Date(m.timestamp) < 30 * 24 * 60 * 60 * 1000
        );
      default:
        return sorted;
    }
  }, [memories, timeRange]);
  
  useEffect(() => {
    if (!filteredMemories.length || !svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const { width, height } = dimensions;
    const margin = { top: 40, right: 40, bottom: 60, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Time scale
    const timeExtent = d3.extent(filteredMemories, d => new Date(d.timestamp));
    const xScale = d3.scaleTime()
      .domain(timeExtent)
      .range([0, innerWidth * zoomLevel]);
    
    // Importance scale (y-axis)
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);
    
    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolatePlasma)
      .domain([0, 1]);
    
    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([1, 10])
      .translateExtent([[0, 0], [innerWidth, innerHeight]])
      .on('zoom', (event) => {
        setZoomLevel(event.transform.k);
        xScale.range([0, innerWidth * event.transform.k]);
        updateVisualization();
      });
    
    svg.call(zoom);
    
    // Add gradient background
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'timeline-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');
    
    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#1a1a1a')
      .attr('stop-opacity', 1);
    
    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#0a0a0a')
      .attr('stop-opacity', 1);
    
    svg.append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'url(#timeline-gradient)');
    
    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.timeFormat('%b %d'))
      .ticks(10);
    
    const yAxis = d3.axisLeft(yScale)
      .tickFormat(d => `${(d * 100).toFixed(0)}%`);
    
    const xAxisGroup = g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis);
    
    g.append('g')
      .attr('class', 'y-axis')
      .call(yAxis);
    
    // Style axes
    g.selectAll('.axis')
      .style('color', '#666');
    
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
    
    // Create line generator
    const line = d3.line()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.importance))
      .curve(d3.curveMonotoneX);
    
    // Add importance line
    g.append('path')
      .datum(filteredMemories)
      .attr('class', 'importance-line')
      .attr('fill', 'none')
      .attr('stroke', '#00ff88')
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Add area under line
    const area = d3.area()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(innerHeight)
      .y1(d => yScale(d.importance))
      .curve(d3.curveMonotoneX);
    
    g.append('path')
      .datum(filteredMemories)
      .attr('class', 'importance-area')
      .attr('fill', 'url(#area-gradient)')
      .attr('opacity', 0.3)
      .attr('d', area);
    
    // Add area gradient
    const areaGradient = svg.select('defs')
      .append('linearGradient')
      .attr('id', 'area-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');
    
    areaGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#00ff88')
      .attr('stop-opacity', 0.8);
    
    areaGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#00ff88')
      .attr('stop-opacity', 0.1);
    
    // Add memory points
    const memoryGroups = g.selectAll('.memory-point')
      .data(filteredMemories)
      .enter().append('g')
      .attr('class', 'memory-point')
      .attr('transform', d => `translate(${xScale(new Date(d.timestamp))},${yScale(d.importance)})`);
    
    // Add circles
    memoryGroups.append('circle')
      .attr('r', d => 4 + d.importance * 6)
      .attr('fill', d => colorScale(d.importance))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', d => selectedId === d.id ? 3 : 1)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        onMemorySelect(d);
      })
      .on('mouseenter', (event, d) => {
        setHoveredMemory({ ...d, x: xScale(new Date(d.timestamp)), y: yScale(d.importance) });
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', d => 6 + d.importance * 8);
      })
      .on('mouseleave', (event, d) => {
        setHoveredMemory(null);
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', d => 4 + d.importance * 6);
      });
    
    // Add pulse animation for selected memory
    if (selectedId) {
      const selectedMemory = filteredMemories.find(m => m.id === selectedId);
      if (selectedMemory) {
        const pulse = g.append('circle')
          .attr('cx', xScale(new Date(selectedMemory.timestamp)))
          .attr('cy', yScale(selectedMemory.importance))
          .attr('r', 10)
          .attr('fill', 'none')
          .attr('stroke', '#00ff88')
          .attr('stroke-width', 2);
        
        const animatePulse = () => {
          pulse
            .attr('r', 10)
            .attr('opacity', 1)
            .transition()
            .duration(1000)
            .attr('r', 20)
            .attr('opacity', 0)
            .on('end', animatePulse);
        };
        
        animatePulse();
      }
    }
    
    // Add time markers for significant events
    const significantMemories = filteredMemories.filter(m => m.importance > 0.8);
    
    g.selectAll('.time-marker')
      .data(significantMemories)
      .enter().append('line')
      .attr('class', 'time-marker')
      .attr('x1', d => xScale(new Date(d.timestamp)))
      .attr('y1', 0)
      .attr('x2', d => xScale(new Date(d.timestamp)))
      .attr('y2', innerHeight)
      .attr('stroke', '#ffaa00')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.3);
    
    const updateVisualization = () => {
      xAxisGroup.call(xAxis);
      
      g.select('.importance-line')
        .attr('d', line);
      
      g.select('.importance-area')
        .attr('d', area);
      
      memoryGroups
        .attr('transform', d => `translate(${xScale(new Date(d.timestamp))},${yScale(d.importance)})`);
      
      g.selectAll('.time-marker')
        .attr('x1', d => xScale(new Date(d.timestamp)))
        .attr('x2', d => xScale(new Date(d.timestamp)));
    };
    
  }, [filteredMemories, dimensions, selectedId, onMemorySelect, zoomLevel]);
  
  return (
    <div ref={containerRef} className="memory-timeline-container">
      <div className="timeline-controls">
        <div className="time-range-selector">
          <button
            className={timeRange === 'hour' ? 'active' : ''}
            onClick={() => setTimeRange('hour')}
          >
            Hour
          </button>
          <button
            className={timeRange === 'day' ? 'active' : ''}
            onClick={() => setTimeRange('day')}
          >
            Day
          </button>
          <button
            className={timeRange === 'week' ? 'active' : ''}
            onClick={() => setTimeRange('week')}
          >
            Week
          </button>
          <button
            className={timeRange === 'month' ? 'active' : ''}
            onClick={() => setTimeRange('month')}
          >
            Month
          </button>
          <button
            className={timeRange === 'all' ? 'active' : ''}
            onClick={() => setTimeRange('all')}
          >
            All
          </button>
        </div>
        
        <div className="zoom-indicator">
          Zoom: {zoomLevel.toFixed(1)}x
        </div>
      </div>
      
      <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />
      
      <AnimatePresence>
        {hoveredMemory && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="timeline-tooltip"
            style={{
              position: 'absolute',
              left: hoveredMemory.x + 60,
              top: hoveredMemory.y + 40,
            }}
          >
            <div className="tooltip-time">
              {new Date(hoveredMemory.timestamp).toLocaleString()}
            </div>
            <div className="tooltip-content">{hoveredMemory.content}</div>
            <div className="tooltip-footer">
              <span className="importance">
                Importance: {(hoveredMemory.importance * 100).toFixed(0)}%
              </span>
              <span className="memory-id">#{hoveredMemory.id}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .memory-timeline-container {
          position: relative;
          width: 100%;
          height: 100%;
          background: #0a0a0a;
          border-radius: 8px;
          overflow: hidden;
        }
        
        .timeline-controls {
          position: absolute;
          top: 16px;
          left: 16px;
          right: 16px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          z-index: 10;
        }
        
        .time-range-selector {
          display: flex;
          gap: 4px;
          background: rgba(26, 26, 26, 0.8);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          padding: 4px;
        }
        
        .time-range-selector button {
          background: none;
          border: none;
          color: #666;
          padding: 6px 12px;
          font-size: 12px;
          cursor: pointer;
          border-radius: 4px;
          transition: all 0.2s;
        }
        
        .time-range-selector button:hover {
          color: #e0e0e0;
          background: rgba(255, 255, 255, 0.05);
        }
        
        .time-range-selector button.active {
          background: rgba(0, 255, 136, 0.2);
          color: #00ff88;
        }
        
        .zoom-indicator {
          background: rgba(26, 26, 26, 0.8);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          padding: 6px 12px;
          font-size: 12px;
          color: #888;
        }
        
        .timeline-tooltip {
          background: rgba(26, 26, 26, 0.95);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 12px;
          max-width: 300px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
          pointer-events: none;
        }
        
        .tooltip-time {
          font-size: 11px;
          color: #00ff88;
          margin-bottom: 6px;
        }
        
        .tooltip-content {
          color: #e0e0e0;
          font-size: 13px;
          line-height: 1.4;
          margin-bottom: 8px;
        }
        
        .tooltip-footer {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: #666;
        }
        
        .importance {
          color: #ffaa00;
        }
      `}</style>
    </div>
  );
};

export default MemoryTimeline;