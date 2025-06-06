import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';
import './MemoryGraph.css';

const MemoryGraph = ({ memories = [], links = [], onMemorySelect, selectedId }) => {
  const svgRef = useRef();
  const containerRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredMemory, setHoveredMemory] = useState(null);
  const simulationRef = useRef(null);
  
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
  
  useEffect(() => {
    if (!memories.length || !svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const { width, height } = dimensions;
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Create container for zoom/pan
    const g = svg.append('g');
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);
    
    // Color scale based on memory importance
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);
    
    // Create force simulation
    const simulation = d3.forceSimulation(memories)
      .force('link', d3.forceLink(links)
        .id(d => d.id)
        .distance(d => 100 * (1 - d.value))
        .strength(d => d.value))
      .force('charge', d3.forceManyBody()
        .strength(d => -200 * d.importance))
      .force('center', d3.forceCenter(centerX, centerY))
      .force('collision', d3.forceCollide()
        .radius(d => 20 + d.importance * 30));
    
    simulationRef.current = simulation;
    
    // Create gradient definitions
    const defs = svg.append('defs');
    
    // Create glow filter
    const filter = defs.append('filter')
      .attr('id', 'glow');
    
    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');
    
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode')
      .attr('in', 'coloredBlur');
    feMerge.append('feMergeNode')
      .attr('in', 'SourceGraphic');
    
    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', '#ffffff')
      .attr('stroke-opacity', d => d.value * 0.3)
      .attr('stroke-width', d => Math.sqrt(d.value) * 3);
    
    // Create node groups
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(memories)
      .enter().append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));
    
    // Add circles with glow effect
    node.append('circle')
      .attr('r', d => 10 + d.importance * 20)
      .attr('fill', d => colorScale(d.importance))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', d => selectedId === d.id ? 3 : 1)
      .attr('filter', d => selectedId === d.id ? 'url(#glow)' : null)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        onMemorySelect(d);
      })
      .on('mouseenter', (event, d) => {
        setHoveredMemory(d);
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', d => 15 + d.importance * 25);
      })
      .on('mouseleave', (event, d) => {
        setHoveredMemory(null);
        d3.select(event.target)
          .transition()
          .duration(200)
          .attr('r', d => 10 + d.importance * 20);
      });
    
    // Add importance rings
    node.append('circle')
      .attr('r', d => 15 + d.importance * 25)
      .attr('fill', 'none')
      .attr('stroke', d => colorScale(d.importance))
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.3)
      .attr('stroke-dasharray', '3,3');
    
    // Add labels for important memories
    node.filter(d => d.importance > 0.7)
      .append('text')
      .text(d => d.content.substring(0, 20) + '...')
      .attr('x', 0)
      .attr('y', d => -20 - d.importance * 20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
      .style('font-size', '10px')
      .style('pointer-events', 'none');
    
    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    
    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    
    // Add initial animation
    svg.style('opacity', 0)
      .transition()
      .duration(500)
      .style('opacity', 1);
    
    // Focus on selected memory
    if (selectedId) {
      const selectedMemory = memories.find(m => m.id === selectedId);
      if (selectedMemory) {
        const scale = 2;
        const transform = d3.zoomIdentity
          .translate(width / 2, height / 2)
          .scale(scale)
          .translate(-selectedMemory.x || 0, -selectedMemory.y || 0);
        
        svg.transition()
          .duration(750)
          .call(zoom.transform, transform);
      }
    }
    
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [memories, links, dimensions, selectedId, onMemorySelect]);
  
  return (
    <div ref={containerRef} className="memory-graph-container">
      <svg ref={svgRef} width={dimensions.width} height={dimensions.height}>
        <rect width="100%" height="100%" fill="#0a0a0a" />
      </svg>
      
      {hoveredMemory && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="memory-tooltip"
          style={{
            position: 'absolute',
            left: hoveredMemory.x,
            top: hoveredMemory.y - 50,
            transform: 'translate(-50%, -100%)'
          }}
        >
          <div className="tooltip-content">
            <div className="tooltip-header">
              <span className="importance-badge" style={{
                backgroundColor: d3.scaleSequential(d3.interpolateViridis)(hoveredMemory.importance)
              }}>
                {(hoveredMemory.importance * 100).toFixed(0)}%
              </span>
              <span className="memory-type">{hoveredMemory.type || 'episodic'}</span>
            </div>
            <p className="memory-preview">{hoveredMemory.content}</p>
            <div className="memory-meta">
              <span>{new Date(hoveredMemory.timestamp).toLocaleDateString()}</span>
              <span>ID: {hoveredMemory.id}</span>
            </div>
          </div>
        </motion.div>
      )}
      
      <div className="graph-controls">
        <button onClick={() => simulationRef.current?.restart()}>
          <span className="icon">🔄</span> Reorganize
        </button>
        <button onClick={() => {
          const svg = d3.select(svgRef.current);
          const zoom = d3.zoom();
          svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }}>
          <span className="icon">🎯</span> Reset View
        </button>
      </div>
      
      <style jsx>{`
        .memory-graph-container {
          position: relative;
          width: 100%;
          height: 100%;
          background: #0a0a0a;
          border-radius: 8px;
          overflow: hidden;
        }
        
        .memory-tooltip {
          background: rgba(26, 26, 26, 0.95);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 12px;
          max-width: 300px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
          z-index: 1000;
          pointer-events: none;
        }
        
        .tooltip-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        
        .importance-badge {
          padding: 2px 8px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 600;
          color: white;
        }
        
        .memory-type {
          font-size: 11px;
          color: #888;
          text-transform: uppercase;
        }
        
        .memory-preview {
          color: #e0e0e0;
          font-size: 13px;
          line-height: 1.4;
          margin: 8px 0;
        }
        
        .memory-meta {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: #666;
        }
        
        .graph-controls {
          position: absolute;
          top: 16px;
          right: 16px;
          display: flex;
          gap: 8px;
        }
        
        .graph-controls button {
          background: rgba(26, 26, 26, 0.8);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          padding: 8px 12px;
          color: #e0e0e0;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .graph-controls button:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.2);
        }
        
        .icon {
          font-size: 16px;
        }
      `}</style>
    </div>
  );
};

export default MemoryGraph;