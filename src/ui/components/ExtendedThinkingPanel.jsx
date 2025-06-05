import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';

const ExtendedThinkingPanel = ({ sessionId, isVisible = false }) => {
  const [thoughts, setThoughts] = useState([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedThought, setSelectedThought] = useState(null);
  const [viewMode, setViewMode] = useState('stream'); // 'stream', 'tree', 'timeline'
  const svgRef = useRef(null);
  const wsRef = useRef(null);
  
  // Color mapping for thought types
  const thoughtColors = {
    analytical: '#2563EB',
    emotional: '#7C3AED',
    creative: '#EA580C',
    uncertain: '#6B7280',
    hypothesis: '#059669',
    validation: '#4F46E5',
    decision: '#DC2626',
    meta: '#DB2777'
  };
  
  // Connect to WebSocket for real-time updates
  useEffect(() => {
    if (!isVisible) return;
    
    const ws = new WebSocket(`ws://localhost:8765?session_id=${sessionId}`);
    wsRef.current = ws;
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'thinking_update') {
        setThoughts(prev => [...prev, ...message.data.thoughts]);
      }
    };
    
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [sessionId, isVisible]);
  
  // Render tree visualization with D3
  useEffect(() => {
    if (viewMode !== 'tree' || thoughts.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    
    // Convert thoughts to hierarchical data
    const root = d3.stratify()
      .id(d => d.id)
      .parentId(d => d.parent_id)(thoughts);
    
    const treeLayout = d3.tree()
      .size([width - margin.left - margin.right, height - margin.top - margin.bottom]);
    
    const treeData = treeLayout(root);
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Draw links
    g.selectAll('.link')
      .data(treeData.links())
      .enter().append('path')
      .attr('class', 'link')
      .attr('d', d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y))
      .attr('fill', 'none')
      .attr('stroke', '#374151')
      .attr('stroke-width', 2)
      .attr('opacity', 0.6);
    
    // Draw nodes
    const nodes = g.selectAll('.node')
      .data(treeData.descendants())
      .enter().append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.x},${d.y})`);
    
    nodes.append('circle')
      .attr('r', 8)
      .attr('fill', d => thoughtColors[d.data.type] || '#6B7280')
      .attr('stroke', '#1F2937')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => setSelectedThought(d.data));
    
    // Add hover effect
    nodes.on('mouseenter', function(event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', 10);
    })
    .on('mouseleave', function(event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', 8);
    });
    
  }, [thoughts, viewMode]);
  
  const renderStreamView = () => (
    <div className="thought-stream">
      {thoughts.map((thought, index) => (
        <div
          key={thought.id}
          className={`thought-item ${thought.type}`}
          style={{
            borderLeft: `4px solid ${thoughtColors[thought.type]}`,
            animation: 'fadeInUp 0.3s ease-out',
            animationDelay: `${index * 0.05}s`,
            animationFillMode: 'both'
          }}
        >
          <div className="thought-header">
            <span 
              className="thought-type"
              style={{ color: thoughtColors[thought.type] }}
            >
              {thought.type.toUpperCase()}
            </span>
            <span className="thought-confidence">
              {Math.round(thought.confidence * 100)}% confident
            </span>
          </div>
          <div className="thought-content">{thought.content}</div>
          {thought.parent_id && (
            <div className="thought-connection">
              ↳ Connected to previous thought
            </div>
          )}
        </div>
      ))}
    </div>
  );
  
  const renderTimelineView = () => (
    <div className="thought-timeline">
      <div className="timeline-line" />
      {thoughts.map((thought, index) => (
        <div
          key={thought.id}
          className={`timeline-item ${index % 2 === 0 ? 'left' : 'right'}`}
        >
          <div
            className="timeline-dot"
            style={{ backgroundColor: thoughtColors[thought.type] }}
          />
          <div className="timeline-content">
            <div className="timeline-time">
              {new Date(thought.timestamp).toLocaleTimeString()}
            </div>
            <div className="timeline-thought">
              <strong>{thought.type}:</strong> {thought.content}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
  
  const renderTreeView = () => (
    <div className="thought-tree">
      <svg ref={svgRef} />
      {selectedThought && (
        <div className="thought-detail">
          <h4>Selected Thought</h4>
          <p><strong>Type:</strong> {selectedThought.type}</p>
          <p><strong>Confidence:</strong> {Math.round(selectedThought.confidence * 100)}%</p>
          <p><strong>Content:</strong> {selectedThought.content}</p>
        </div>
      )}
    </div>
  );
  
  if (!isVisible) return null;
  
  return (
    <div className={`extended-thinking-panel ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <div className="panel-header">
        <h3>
          <span className="thinking-indicator">
            <span className="pulse-dot" />
            Extended Thinking
          </span>
        </h3>
        <div className="panel-controls">
          <div className="view-switcher">
            <button
              className={viewMode === 'stream' ? 'active' : ''}
              onClick={() => setViewMode('stream')}
            >
              Stream
            </button>
            <button
              className={viewMode === 'tree' ? 'active' : ''}
              onClick={() => setViewMode('tree')}
            >
              Tree
            </button>
            <button
              className={viewMode === 'timeline' ? 'active' : ''}
              onClick={() => setViewMode('timeline')}
            >
              Timeline
            </button>
          </div>
          <button
            className="expand-button"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? '⇲' : '⇱'}
          </button>
        </div>
      </div>
      
      <div className="panel-content">
        {thoughts.length === 0 ? (
          <div className="empty-state">
            <div className="loading-thoughts">
              <div className="thought-bubble" />
              <div className="thought-bubble" />
              <div className="thought-bubble" />
            </div>
            <p>Gathering thoughts...</p>
          </div>
        ) : (
          <>
            {viewMode === 'stream' && renderStreamView()}
            {viewMode === 'tree' && renderTreeView()}
            {viewMode === 'timeline' && renderTimelineView()}
          </>
        )}
      </div>
      
      <style jsx>{`
        .extended-thinking-panel {
          position: fixed;
          right: 20px;
          bottom: 20px;
          width: 400px;
          max-height: 600px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 12px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
          transition: all 0.3s ease;
          z-index: 1000;
        }
        
        .extended-thinking-panel.expanded {
          width: 800px;
          max-height: 80vh;
        }
        
        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px;
          border-bottom: 1px solid #333;
        }
        
        .thinking-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .pulse-dot {
          width: 8px;
          height: 8px;
          background: #10b981;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.2); }
          100% { opacity: 1; transform: scale(1); }
        }
        
        .panel-controls {
          display: flex;
          gap: 12px;
          align-items: center;
        }
        
        .view-switcher {
          display: flex;
          gap: 4px;
          background: #1a1a1a;
          padding: 4px;
          border-radius: 6px;
        }
        
        .view-switcher button {
          padding: 4px 12px;
          background: none;
          border: none;
          color: #666;
          font-size: 12px;
          cursor: pointer;
          border-radius: 4px;
          transition: all 0.2s;
        }
        
        .view-switcher button:hover {
          color: #e0e0e0;
        }
        
        .view-switcher button.active {
          background: #333;
          color: #e0e0e0;
        }
        
        .panel-content {
          padding: 16px;
          max-height: calc(100% - 60px);
          overflow-y: auto;
        }
        
        .thought-stream {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        
        .thought-item {
          padding: 12px;
          background: #1a1a1a;
          border-radius: 8px;
          border-left-width: 4px;
          border-left-style: solid;
        }
        
        .thought-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
          font-size: 12px;
        }
        
        .thought-type {
          font-weight: 600;
          letter-spacing: 0.5px;
        }
        
        .thought-confidence {
          color: #888;
        }
        
        .thought-content {
          color: #e0e0e0;
          line-height: 1.5;
        }
        
        .thought-connection {
          margin-top: 8px;
          font-size: 12px;
          color: #666;
          font-style: italic;
        }
        
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .thought-tree {
          position: relative;
        }
        
        .thought-detail {
          position: absolute;
          top: 10px;
          right: 10px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          padding: 12px;
          max-width: 250px;
        }
        
        .timeline-line {
          position: absolute;
          left: 50%;
          top: 0;
          bottom: 0;
          width: 2px;
          background: #333;
        }
        
        .timeline-item {
          position: relative;
          margin-bottom: 24px;
          display: flex;
          align-items: center;
        }
        
        .timeline-item.left {
          justify-content: flex-start;
          padding-right: 50%;
        }
        
        .timeline-item.right {
          justify-content: flex-end;
          padding-left: 50%;
        }
        
        .timeline-dot {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          width: 12px;
          height: 12px;
          border-radius: 50%;
          border: 2px solid #0a0a0a;
        }
        
        .timeline-content {
          background: #1a1a1a;
          padding: 12px;
          border-radius: 8px;
          max-width: 80%;
        }
        
        .timeline-time {
          font-size: 11px;
          color: #666;
          margin-bottom: 4px;
        }
        
        .empty-state {
          text-align: center;
          padding: 40px;
          color: #666;
        }
        
        .loading-thoughts {
          display: flex;
          justify-content: center;
          gap: 8px;
          margin-bottom: 16px;
        }
        
        .thought-bubble {
          width: 12px;
          height: 12px;
          background: #333;
          border-radius: 50%;
          animation: thoughtBubble 1.5s infinite;
        }
        
        .thought-bubble:nth-child(2) {
          animation-delay: 0.2s;
        }
        
        .thought-bubble:nth-child(3) {
          animation-delay: 0.4s;
        }
        
        @keyframes thoughtBubble {
          0%, 100% { transform: scale(0.8); opacity: 0.5; }
          50% { transform: scale(1.2); opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default ExtendedThinkingPanel;