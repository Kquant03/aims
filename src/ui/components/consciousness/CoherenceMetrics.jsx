import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion, AnimatePresence } from 'framer-motion';

const CoherenceMetrics = ({ currentCoherence = 0.7, history = [], evolutionEvents = [] }) => {
  const chartRef = useRef(null);
  const distributionRef = useRef(null);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [timeRange, setTimeRange] = useState('hour');
  
  // Calculate coherence statistics
  const getCoherenceStats = () => {
    if (history.length === 0) {
      return {
        average: currentCoherence,
        min: currentCoherence,
        max: currentCoherence,
        trend: 0,
        volatility: 0
      };
    }
    
    const values = history.map(h => h.coherence);
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    // Calculate trend
    const recentValues = values.slice(-10);
    const olderValues = values.slice(-20, -10);
    const recentAvg = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
    const olderAvg = olderValues.length > 0 
      ? olderValues.reduce((a, b) => a + b, 0) / olderValues.length 
      : recentAvg;
    const trend = recentAvg - olderAvg;
    
    // Calculate volatility
    const variance = values.reduce((sum, val) => sum + Math.pow(val - average, 2), 0) / values.length;
    const volatility = Math.sqrt(variance);
    
    return { average, min, max, trend, volatility };
  };
  
  const stats = getCoherenceStats();
  
  // Draw main coherence chart
  useEffect(() => {
    if (!chartRef.current) return;
    
    const svg = d3.select(chartRef.current);
    svg.selectAll('*').remove();
    
    const width = 600;
    const height = 300;
    const margin = { top: 20, right: 60, bottom: 40, left: 60 };
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Prepare data
    const now = new Date();
    const historyData = history.length > 0 ? history : [
      { timestamp: now, coherence: currentCoherence }
    ];
    
    // Time scale based on selected range
    const timeRanges = {
      hour: 60 * 60 * 1000,
      day: 24 * 60 * 60 * 1000,
      week: 7 * 24 * 60 * 60 * 1000
    };
    
    const rangeMs = timeRanges[timeRange];
    const startTime = new Date(now.getTime() - rangeMs);
    
    const filteredData = historyData.filter(d => 
      new Date(d.timestamp) >= startTime
    );
    
    // Scales
    const xScale = d3.scaleTime()
      .domain([startTime, now])
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);
    
    // Create gradient
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'coherence-gradient')
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
    
    // Area generator
    const area = d3.area()
      .x(d => xScale(new Date(d.timestamp)))
      .y0(innerHeight)
      .y1(d => yScale(d.coherence))
      .curve(d3.curveMonotoneX);
    
    // Line generator
    const line = d3.line()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d.coherence))
      .curve(d3.curveMonotoneX);
    
    // Add area
    g.append('path')
      .datum(filteredData)
      .attr('fill', 'url(#coherence-gradient)')
      .attr('d', area);
    
    // Add line
    g.append('path')
      .datum(filteredData)
      .attr('fill', 'none')
      .attr('stroke', '#00a8ff')
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Add threshold lines
    const thresholds = [
      { value: 0.8, label: 'High', color: '#00ff88' },
      { value: 0.6, label: 'Medium', color: '#ffaa00' },
      { value: 0.4, label: 'Low', color: '#ff6600' }
    ];
    
    thresholds.forEach(threshold => {
      g.append('line')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', yScale(threshold.value))
        .attr('y2', yScale(threshold.value))
        .attr('stroke', threshold.color)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0.5);
      
      g.append('text')
        .attr('x', innerWidth + 5)
        .attr('y', yScale(threshold.value))
        .attr('dy', '0.3em')
        .attr('fill', threshold.color)
        .attr('font-size', '11px')
        .text(threshold.label);
    });
    
    // Add evolution events
    const eventMarkers = g.selectAll('.event-marker')
      .data(evolutionEvents.filter(e => new Date(e.timestamp) >= startTime))
      .enter()
      .append('g')
      .attr('class', 'event-marker')
      .attr('transform', d => `translate(${xScale(new Date(d.timestamp))}, 0)`);
    
    eventMarkers.append('line')
      .attr('y1', 0)
      .attr('y2', innerHeight)
      .attr('stroke', '#ff00aa')
      .attr('stroke-width', 2)
      .attr('opacity', 0.5);
    
    eventMarkers.append('circle')
      .attr('cy', d => {
        const dataPoint = filteredData.find(h => 
          Math.abs(new Date(h.timestamp) - new Date(d.timestamp)) < 60000
        );
        return dataPoint ? yScale(dataPoint.coherence) : yScale(0.5);
      })
      .attr('r', 5)
      .attr('fill', '#ff00aa')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('cursor', 'pointer')
      .on('click', (event, d) => setSelectedEvent(d));
    
    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')));
    
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(d => `${d * 100}%`));
    
    // Add current value indicator
    if (filteredData.length > 0) {
      const lastPoint = filteredData[filteredData.length - 1];
      const x = xScale(new Date(lastPoint.timestamp));
      const y = yScale(lastPoint.coherence);
      
      g.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 6)
        .attr('fill', '#00a8ff')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);
      
      g.append('text')
        .attr('x', x)
        .attr('y', y - 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#00a8ff')
        .attr('font-weight', 'bold')
        .attr('font-size', '14px')
        .text(`${(lastPoint.coherence * 100).toFixed(1)}%`);
    }
    
  }, [history, timeRange, evolutionEvents, currentCoherence]);
  
  // Draw coherence distribution
  useEffect(() => {
    if (!distributionRef.current || history.length < 10) return;
    
    const svg = d3.select(distributionRef.current);
    svg.selectAll('*').remove();
    
    const width = 200;
    const height = 150;
    const margin = { top: 10, right: 10, bottom: 30, left: 30 };
    
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create histogram
    const values = history.map(h => h.coherence);
    
    const x = d3.scaleLinear()
      .domain([0, 1])
      .range([0, innerWidth]);
    
    const histogram = d3.histogram()
      .domain(x.domain())
      .thresholds(x.ticks(20));
    
    const bins = histogram(values);
    
    const y = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length)])
      .range([innerHeight, 0]);
    
    // Draw bars
    g.selectAll('.bar')
      .data(bins)
      .enter()
      .append('rect')
      .attr('x', d => x(d.x0))
      .attr('y', d => y(d.length))
      .attr('width', d => x(d.x1) - x(d.x0) - 1)
      .attr('height', d => innerHeight - y(d.length))
      .attr('fill', '#00a8ff')
      .attr('opacity', 0.7);
    
    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d => `${d * 100}%`));
    
    g.append('g')
      .call(d3.axisLeft(y).ticks(5));
    
  }, [history]);
  
  const getCoherenceLevel = (value) => {
    if (value > 0.8) return { label: 'Excellent', color: '#00ff88' };
    if (value > 0.6) return { label: 'Good', color: '#00a8ff' };
    if (value > 0.4) return { label: 'Fair', color: '#ffaa00' };
    return { label: 'Low', color: '#ff6600' };
  };
  
  const coherenceLevel = getCoherenceLevel(currentCoherence);
  
  return (
    <div className="coherence-metrics">
      <div className="metrics-header">
        <h3>Coherence Metrics</h3>
        <div className="time-range-selector">
          {['hour', 'day', 'week'].map(range => (
            <button
              key={range}
              className={timeRange === range ? 'active' : ''}
              onClick={() => setTimeRange(range)}
            >
              {range.charAt(0).toUpperCase() + range.slice(1)}
            </button>
          ))}
        </div>
      </div>
      
      <div className="current-coherence">
        <div className="coherence-display">
          <div className="coherence-value" style={{ color: coherenceLevel.color }}>
            {(currentCoherence * 100).toFixed(1)}%
          </div>
          <div className="coherence-label">{coherenceLevel.label}</div>
        </div>
        
        <div className="coherence-ring">
          <svg width="120" height="120">
            <circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke="#1a1a1a"
              strokeWidth="10"
            />
            <circle
              cx="60"
              cy="60"
              r="50"
              fill="none"
              stroke={coherenceLevel.color}
              strokeWidth="10"
              strokeDasharray={`${currentCoherence * 314} 314`}
              strokeDashoffset="0"
              transform="rotate(-90 60 60)"
              style={{ transition: 'stroke-dasharray 0.5s ease' }}
            />
          </svg>
        </div>
      </div>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <label>Average</label>
          <span>{(stats.average * 100).toFixed(1)}%</span>
        </div>
        <div className="metric-card">
          <label>Range</label>
          <span>{(stats.min * 100).toFixed(0)}-{(stats.max * 100).toFixed(0)}%</span>
        </div>
        <div className="metric-card">
          <label>Trend</label>
          <span className={stats.trend > 0 ? 'positive' : stats.trend < 0 ? 'negative' : 'neutral'}>
            {stats.trend > 0 ? '↑' : stats.trend < 0 ? '↓' : '→'} 
            {Math.abs(stats.trend * 100).toFixed(1)}%
          </span>
        </div>
        <div className="metric-card">
          <label>Volatility</label>
          <span>{(stats.volatility * 100).toFixed(1)}%</span>
        </div>
      </div>
      
      <div className="chart-section">
        <h4>Coherence Over Time</h4>
        <svg ref={chartRef} />
      </div>
      
      {history.length >= 10 && (
        <div className="distribution-section">
          <h4>Distribution</h4>
          <svg ref={distributionRef} />
        </div>
      )}
      
      <AnimatePresence>
        {selectedEvent && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="event-detail"
          >
            <div className="event-header">
              <h4>Evolution Event</h4>
              <button onClick={() => setSelectedEvent(null)}>×</button>
            </div>
            <div className="event-content">
              <p><strong>Type:</strong> {selectedEvent.type}</p>
              <p><strong>Time:</strong> {new Date(selectedEvent.timestamp).toLocaleString()}</p>
              <p><strong>Impact:</strong> {(selectedEvent.impact * 100).toFixed(0)}%</p>
              <p><strong>Description:</strong> {selectedEvent.description}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .coherence-metrics {
          padding: 20px;
          background: #0a0a0a;
          border-radius: 12px;
          border: 1px solid #333;
        }
        
        .metrics-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }
        
        .metrics-header h3 {
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
        
        .time-range-selector button {
          padding: 6px 12px;
          background: none;
          border: none;
          color: #666;
          font-size: 12px;
          cursor: pointer;
          border-radius: 4px;
          transition: all 0.2s;
        }
        
        .time-range-selector button:hover {
          color: #e0e0e0;
        }
        
        .time-range-selector button.active {
          background: #333;
          color: #e0e0e0;
        }
        
        .current-coherence {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
          padding: 20px;
          background: #1a1a1a;
          border-radius: 12px;
        }
        
        .coherence-display {
          text-align: center;
        }
        
        .coherence-value {
          font-size: 48px;
          font-weight: 700;
          line-height: 1;
          margin-bottom: 8px;
        }
        
        .coherence-label {
          font-size: 16px;
          color: #888;
          font-weight: 600;
        }
        
        .coherence-ring {
          position: relative;
        }
        
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
          gap: 12px;
          margin-bottom: 24px;
        }
        
        .metric-card {
          padding: 16px;
          background: #1a1a1a;
          border-radius: 8px;
          text-align: center;
        }
        
        .metric-card label {
          display: block;
          font-size: 12px;
          color: #666;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 8px;
        }
        
        .metric-card span {
          display: block;
          font-size: 20px;
          font-weight: 600;
          color: #e0e0e0;
        }
        
        .metric-card .positive {
          color: #00ff88;
        }
        
        .metric-card .negative {
          color: #ff6600;
        }
        
        .metric-card .neutral {
          color: #888;
        }
        
        .chart-section, .distribution-section {
          margin-top: 24px;
        }
        
        .chart-section h4, .distribution-section h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
          color: #888;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .event-detail {
          position: fixed;
          bottom: 20px;
          right: 20px;
          width: 320px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
          z-index: 1000;
        }
        
        .event-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px;
          border-bottom: 1px solid #333;
        }
        
        .event-header h4 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
        }
        
        .event-header button {
          width: 24px;
          height: 24px;
          background: none;
          border: none;
          color: #666;
          font-size: 20px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .event-header button:hover {
          color: #e0e0e0;
        }
        
        .event-content {
          padding: 16px;
        }
        
        .event-content p {
          margin: 0 0 8px 0;
          font-size: 14px;
          color: #a0a0a0;
        }
        
        .event-content strong {
          color: #e0e0e0;
        }
      `}</style>
    </div>
  );
};

export default CoherenceMetrics;