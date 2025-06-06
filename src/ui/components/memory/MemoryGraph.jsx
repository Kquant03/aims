import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';

const MemoryGraph = ({ memories, links, onMemorySelect, selectedId }) => {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const simulationRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredMemory, setHoveredMemory] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  
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
  
  // Draw the graph
  useEffect(() => {
    if (!memories || memories.length === 0 || !svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const { width, height } = dimensions;
    
    // Create main group for zoom/pan
    const g = svg.append('g');
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoomLevel(event.transform.k);
      });
    
    svg.call(zoom);
    
    // Add gradient definitions
    const defs = svg.append('defs');
    
    // Create gradients for different importance levels
    const importanceGradients = [
      { id: 'high-importance', color1: '#ff0066', color2: '#ff3388' },
      { id: 'medium-importance', color1: '#00a8ff', color2: '#33bbff' },
      { id: 'low-importance', color1: '#00ff88', color2: '#33ff99' }
    ];
    
    importanceGradients.forEach(grad => {
      const gradient = defs.append('radialGradient')
        .attr('id', grad.id);
      
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', grad.color1)
        .attr('stop-opacity', 0.8);
      
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', grad.color2)
        .attr('stop-opacity', 0.3);
    });
    
    // Process data
    const nodes = memories.map(memory => ({
      ...memory,
      x: width / 2 + (Math.random() - 0.5) * 200,
      y: height / 2 + (Math.random() - 0.5) * 200,
      r: 5 + memory.importance * 25
    }));
    
    const validLinks = links.filter(link => {
      const sourceExists = nodes.find(n => n.id === link.source);
      const targetExists = nodes.find(n => n.id === link.target);
      return sourceExists && targetExists;
    });
    
    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(validLinks)
        .id(d => d.id)
        .distance(d => 50 + (1 - d.value) * 50)
        .strength(d => d.value * 0.5))
      .force('charge', d3.forceManyBody()
        .strength(d => -100 - d.importance * 100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide()
        .radius(d => d.r + 5));
    
    simulationRef.current = simulation;
    
    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(validLinks)
      .enter().append('line')
      .attr('stroke', '#ffffff')
      .attr('stroke-opacity', d => d.value * 0.3)
      .attr('stroke-width', d => Math.sqrt(d.value) * 2);
    
    // Create node groups
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .enter().append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));
    
    // Add circles
    node.append('circle')
      .attr('r', d => d.r)
      .attr('fill', d => {
        if (d.importance > 0.7) return 'url(#high-importance)';
        if (d.importance > 0.4) return 'url(#medium-importance)';
        return 'url(#low-importance)';
      })
      .attr('stroke', d => d.id === selectedId ? '#fff' : 'none')
      .attr('stroke-width', 3)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        onMemorySelect(d);
      })
      .on('mouseover', (event, d) => {
        setHoveredMemory(d);
        
        // Highlight connected nodes
        link.style('stroke-opacity', l => 
          l.source.id === d.id || l.target.id === d.id ? 0.8 : 0.1
        );
        
        node.style('opacity', n => {
          if (n.id === d.id) return 1;
          const connected = validLinks.some(l => 
            (l.source.id === d.id && l.target.id === n.id) ||
            (l.target.id === d.id && l.source.id === n.id)
          );
          return connected ? 1 : 0.3;
        });
      })
      .on('mouseout', () => {
        setHoveredMemory(null);
        link.style('stroke-opacity', d => d.value * 0.3);
        node.style('opacity', 1);
      });
    
    // Add labels (shown on hover or when zoomed in)
    const labels = node.append('text')
      .text(d => d.content.substring(0, 30) + '...')
      .attr('text-anchor', 'middle')
      .attr('dy', d => d.r + 15)
      .style('font-size', '12px')
      .style('fill', '#e0e0e0')
      .style('pointer-events', 'none')
      .style('opacity', 0);
    
    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      
      node.attr('transform', d => `translate(${d.x},${d.y})`);
      
      // Show labels when zoomed in
      labels.style('opacity', zoomLevel > 2 ? 1 : 0);
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
    
    // Add reset zoom button handler
    svg.on('dblclick.zoom', null);
    
    // Cleanup
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [memories, links, dimensions, selectedId, onMemorySelect]);
  
  const resetView = () => {
    const svg = d3.select(svgRef.current);
    svg.transition()
      .duration(750)
      .call(
        d3.zoom().transform,
        d3.zoomIdentity
      );
  };
  
  const getImportanceLabel = (importance) => {
    if (importance > 0.7) return 'Critical';
    if (importance > 0.4) return 'Important';
    return 'Standard';
  };
  
  return (
    <div className="memory-graph" ref={containerRef}>
      <svg 
        ref={svgRef} 
        width={dimensions.width} 
        height={dimensions.height}
        className="graph-svg"
      />
      
      <div className="graph-controls">
        <button onClick={resetView} className="control-button" title="Reset view">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M2 8C2 11.314 4.686 14 8 14C11.314 14 14 11.314 14 8C14 4.686 11.314 2 8 2C5.5 2 3.5 3.5 2.5 5.5M2.5 2V5.5H6" 
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        
        <div className="zoom-indicator">
          Zoom: {(zoomLevel * 100).toFixed(0)}%
        </div>
      </div>
      
      <motion.div 
        className="graph-legend"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <h4>Memory Importance</h4>
        <div className="legend-item">
          <div className="legend-color high" />
          <span>Critical (70%+)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color medium" />
          <span>Important (40-70%)</span>
        </div>
        <div className="legend-item">
          <div className="legend-color low" />
          <span>Standard (&lt;40%)</span>
        </div>
      </motion.div>
      
      {hoveredMemory && (
        <motion.div 
          className="memory-tooltip"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            left: hoveredMemory.x + 20,
            top: hoveredMemory.y - 20
          }}
        >
          <div className="tooltip-header">
            <span className="tooltip-type">{hoveredMemory.type || 'Memory'}</span>
            <span className="tooltip-importance">
              {getImportanceLabel(hoveredMemory.importance)}
            </span>
          </div>
          <p className="tooltip-content">{hoveredMemory.content}</p>
          <div className="tooltip-meta">
            <span>Importance: {(hoveredMemory.importance * 100).toFixed(0)}%</span>
            <span>{new Date(hoveredMemory.timestamp).toLocaleDateString()}</span>
          </div>
        </motion.div>
      )}
      
      <style jsx>{`
        .memory-graph {
          position: relative;
          width: 100%;
          height: 600px;
          background: #0a0a0a;
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid #333;
        }
        
        .graph-svg {
          cursor: grab;
        }
        
        .graph-svg:active {
          cursor: grabbing;
        }
        
        .graph-controls {
          position: absolute;
          top: 20px;
          right: 20px;
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .control-button {
          width: 36px;
          height: 36px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(26, 26, 26, 0.9);
          border: 1px solid #333;
          border-radius: 8px;
          color: #666;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .control-button:hover {
          background: rgba(37, 37, 37, 0.9);
          color: #e0e0e0;
          border-color: #444;
        }
        
        .zoom-indicator {
          padding: 8px 12px;
          background: rgba(26, 26, 26, 0.9);
          border: 1px solid #333;
          border-radius: 6px;
          font-size: 12px;
          color: #888;
        }
        
        .graph-legend {
          position: absolute;
          bottom: 20px;
          left: 20px;
          padding: 16px;
          background: rgba(26, 26, 26, 0.9);
          border: 1px solid #333;
          border-radius: 8px;
          backdrop-filter: blur(10px);
        }
        
        .graph-legend h4 {
          margin: 0 0 12px 0;
          font-size: 12px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
          font-size: 12px;
          color: #a0a0a0;
        }
        
        .legend-item:last-child {
          margin-bottom: 0;
        }
        
        .legend-color {
          width: 16px;
          height: 16px;
          border-radius: 50%;
        }
        
        .legend-color.high {
          background: radial-gradient(circle, #ff0066 0%, #ff3388 100%);
        }
        
        .legend-color.medium {
          background: radial-gradient(circle, #00a8ff 0%, #33bbff 100%);
        }
        
        .legend-color.low {
          background: radial-gradient(circle, #00ff88 0%, #33ff99 100%);
        }
        
        .memory-tooltip {
          position: absolute;
          max-width: 300px;
          padding: 12px;
          background: rgba(26, 26, 26, 0.95);
          border: 1px solid #333;
          border-radius: 8px;
          backdrop-filter: blur(10px);
          pointer-events: none;
          z-index: 100;
        }
        
        .tooltip-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
          font-size: 12px;
        }
        
        .tooltip-type {
          color: #00a8ff;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .tooltip-importance {
          color: #888;
          font-weight: 600;
        }
        
        .tooltip-content {
          margin: 0 0 8px 0;
          font-size: 13px;
          color: #e0e0e0;
          line-height: 1.5;
        }
        
        .tooltip-meta {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: #666;
        }
        
        /* D3 styles */
        :global(.memory-graph .node) {
          transition: opacity 0.2s;
        }
        
        :global(.memory-graph .links line) {
          transition: stroke-opacity 0.2s;
        }
      `}</style>
    </div>
  );
};

export default MemoryGraph;