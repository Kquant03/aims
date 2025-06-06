import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere, Box } from '@react-three/drei';
import * as d3 from 'd3';

// Consciousness Core 3D Visualization
const ConsciousnessCore3D = ({ coherence, emotionalState, workingMemory }) => {
  const meshRef = useRef();
  const particlesRef = useRef();
  
  // Create particle system for neurons
  const particleCount = 1000;
  const positions = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      // Sphere distribution
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = 2 + Math.random() * 0.5;
      
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
    }
    return pos;
  }, []);
  
  // Animate core based on consciousness state
  useFrame((state) => {
    if (meshRef.current) {
      // Pulsing based on coherence
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1 * coherence;
      meshRef.current.scale.set(scale, scale, scale);
      
      // Rotation based on emotional arousal
      meshRef.current.rotation.y += emotionalState.arousal * 0.01;
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime) * 0.1;
    }
    
    if (particlesRef.current) {
      // Particle movement based on working memory activity
      const positions = particlesRef.current.geometry.attributes.position;
      const time = state.clock.elapsedTime;
      
      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        const x = positions.array[i3];
        const y = positions.array[i3 + 1];
        const z = positions.array[i3 + 2];
        
        // Orbital motion
        const angle = time * 0.1 + i * 0.01;
        positions.array[i3] = x * Math.cos(angle) - z * Math.sin(angle);
        positions.array[i3 + 2] = x * Math.sin(angle) + z * Math.cos(angle);
        
        // Vertical oscillation based on memory
        positions.array[i3 + 1] = y + Math.sin(time * 2 + i) * 0.02 * workingMemory;
      }
      positions.needsUpdate = true;
    }
  });
  
  // Calculate color based on emotional state
  const coreColor = new THREE.Color(
    emotionalState.pleasure,
    1 - emotionalState.arousal,
    emotionalState.dominance
  );
  
  return (
    <group>
      {/* Central consciousness core */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial
          color={coreColor}
          emissive={coreColor}
          emissiveIntensity={coherence * 0.5}
          roughness={0.3}
          metalness={0.7}
        />
      </mesh>
      
      {/* Particle system representing neurons */}
      <points ref={particlesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particleCount}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.05}
          color="#00ff88"
          transparent
          opacity={0.6}
          sizeAttenuation
        />
      </points>
      
      {/* Coherence field */}
      <mesh>
        <sphereGeometry args={[3, 32, 32]} />
        <meshBasicMaterial
          color="#0066ff"
          transparent
          opacity={coherence * 0.1}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
};

// Memory Constellation Network
const MemoryConstellation = ({ memories, connections }) => {
  const groupRef = useRef();
  const [hoveredMemory, setHoveredMemory] = useState(null);
  
  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.001;
    }
  });
  
  return (
    <group ref={groupRef}>
      {/* Memory nodes */}
      {memories.map((memory, i) => (
        <Sphere
          key={memory.id}
          position={memory.position}
          args={[memory.importance * 0.3, 16, 16]}
          onPointerOver={() => setHoveredMemory(memory)}
          onPointerOut={() => setHoveredMemory(null)}
        >
          <meshStandardMaterial
            color={memory.color || '#00ff88'}
            emissive={memory.color || '#00ff88'}
            emissiveIntensity={hoveredMemory?.id === memory.id ? 0.8 : 0.3}
          />
        </Sphere>
      ))}
      
      {/* Connections between memories */}
      {connections.map((connection, i) => (
        <Line
          key={i}
          points={[connection.from, connection.to]}
          color="#ffffff"
          lineWidth={connection.strength * 2}
          transparent
          opacity={0.3}
        />
      ))}
      
      {/* Hovered memory label */}
      {hoveredMemory && (
        <Text
          position={[hoveredMemory.position[0], hoveredMemory.position[1] + 0.5, hoveredMemory.position[2]]}
          fontSize={0.2}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {hoveredMemory.content.substring(0, 50)}...
        </Text>
      )}
    </group>
  );
};

// Force-Directed Memory Graph with D3
const MemoryForceGraph = ({ memories, links, width = 800, height = 600 }) => {
  const svgRef = useRef();
  const simulationRef = useRef();
  
  useEffect(() => {
    if (!memories || memories.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    // Create force simulation
    const simulation = d3.forceSimulation(memories)
      .force('link', d3.forceLink(links).id(d => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => d.importance * 20 + 5));
    
    simulationRef.current = simulation;
    
    // Create container
    const g = svg.append('g');
    
    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);
    
    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .enter().append('line')
      .attr('stroke', '#ffffff')
      .attr('stroke-opacity', 0.3)
      .attr('stroke-width', d => Math.sqrt(d.value));
    
    // Create nodes
    const node = g.append('g')
      .selectAll('g')
      .data(memories)
      .enter().append('g')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));
    
    // Add circles
    node.append('circle')
      .attr('r', d => d.importance * 20 + 5)
      .attr('fill', d => d.color || '#00ff88')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer');
    
    // Add labels
    node.append('text')
      .text(d => d.content.substring(0, 20))
      .attr('x', 0)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .style('font-size', '10px')
      .style('pointer-events', 'none');
    
    // Add tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'memory-tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '10px')
      .style('border-radius', '5px')
      .style('font-size', '12px');
    
    node.on('mouseover', (event, d) => {
      tooltip.transition().duration(200).style('opacity', .9);
      tooltip.html(`
        <strong>Memory ${d.id}</strong><br/>
        Content: ${d.content}<br/>
        Importance: ${d.importance.toFixed(2)}<br/>
        Type: ${d.type}
      `)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseout', () => {
      tooltip.transition().duration(500).style('opacity', 0);
    });
    
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
    
    // Cleanup
    return () => {
      simulation.stop();
      tooltip.remove();
    };
  }, [memories, links, width, height]);
  
  return (
    <svg ref={svgRef} width={width} height={height}>
      <defs>
        <radialGradient id="memoryGradient">
          <stop offset="0%" stopColor="#00ff88" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#00ff88" stopOpacity="0.1" />
        </radialGradient>
      </defs>
    </svg>
  );
};

// Emotional Landscape Visualization
const EmotionalLandscape = ({ emotionHistory, width = 800, height = 400 }) => {
  const svgRef = useRef();
  
  useEffect(() => {
    if (!emotionHistory || emotionHistory.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(emotionHistory, d => d.timestamp))
      .range([0, innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);
    
    // Area generators
    const pleasureArea = d3.area()
      .x(d => xScale(d.timestamp))
      .y0(innerHeight)
      .y1(d => yScale(d.pleasure))
      .curve(d3.curveMonotoneX);
    
    const arousalArea = d3.area()
      .x(d => xScale(d.timestamp))
      .y0(innerHeight)
      .y1(d => yScale(d.arousal))
      .curve(d3.curveMonotoneX);
    
    const dominanceArea = d3.area()
      .x(d => xScale(d.timestamp))
      .y0(innerHeight)
      .y1(d => yScale(d.dominance))
      .curve(d3.curveMonotoneX);
    
    // Add areas
    g.append('path')
      .datum(emotionHistory)
      .attr('fill', '#ff6b6b')
      .attr('opacity', 0.3)
      .attr('d', pleasureArea);
    
    g.append('path')
      .datum(emotionHistory)
      .attr('fill', '#4ecdc4')
      .attr('opacity', 0.3)
      .attr('d', arousalArea);
    
    g.append('path')
      .datum(emotionHistory)
      .attr('fill', '#45b7d1')
      .attr('opacity', 0.3)
      .attr('d', dominanceArea);
    
    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')));
    
    g.append('g')
      .call(d3.axisLeft(yScale));
    
    // Add legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth - 100}, 20)`);
    
    const legendData = [
      { label: 'Pleasure', color: '#ff6b6b' },
      { label: 'Arousal', color: '#4ecdc4' },
      { label: 'Dominance', color: '#45b7d1' }
    ];
    
    legendData.forEach((d, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendRow.append('rect')
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', d.color);
      
      legendRow.append('text')
        .attr('x', 15)
        .attr('y', 10)
        .attr('text-anchor', 'start')
        .style('font-size', '12px')
        .style('fill', 'white')
        .text(d.label);
    });
    
  }, [emotionHistory, width, height]);
  
  return <svg ref={svgRef} width={width} height={height} />;
};

// Evolution Timeline Component
const EvolutionTimeline = ({ events, width = 800, height = 200 }) => {
  const svgRef = useRef();
  
  useEffect(() => {
    if (!events || events.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const margin = { top: 20, right: 20, bottom: 40, left: 20 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Scale
    const xScale = d3.scaleTime()
      .domain(d3.extent(events, d => d.timestamp))
      .range([0, innerWidth]);
    
    // Timeline line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', innerHeight / 2)
      .attr('y2', innerHeight / 2)
      .attr('stroke', '#666')
      .attr('stroke-width', 2);
    
    // Event circles
    const eventGroups = g.selectAll('.event')
      .data(events)
      .enter().append('g')
      .attr('class', 'event')
      .attr('transform', d => `translate(${xScale(d.timestamp)}, ${innerHeight / 2})`);
    
    eventGroups.append('circle')
      .attr('r', d => d.impact * 10 + 5)
      .attr('fill', d => {
        const typeColors = {
          'growth': '#00ff88',
          'insight': '#00a8ff',
          'emotional_shift': '#ff00ff',
          'memory_consolidation': '#ffaa00'
        };
        return typeColors[d.type] || '#666';
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer');
    
    // Event labels
    eventGroups.append('text')
      .attr('y', d => (events.indexOf(d) % 2 === 0 ? -20 : 30))
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', 'white')
      .text(d => d.description.substring(0, 20));
    
    // X-axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%H:%M')));
    
  }, [events, width, height]);
  
  return <svg ref={svgRef} width={width} height={height} />;
};

// Main Enhanced Dashboard Component
const EnhancedConsciousnessDashboard = ({ consciousnessData }) => {
  const [view3D, setView3D] = useState(true);
  const [selectedVisualization, setSelectedVisualization] = useState('core');
  
  // Process consciousness data for visualizations
  const processedMemories = useMemo(() => {
    if (!consciousnessData?.memories) return [];
    
    return consciousnessData.memories.map((mem, i) => ({
      ...mem,
      x: Math.random() * 400 - 200,
      y: Math.random() * 400 - 200,
      position: [
        Math.random() * 6 - 3,
        Math.random() * 6 - 3,
        Math.random() * 6 - 3
      ],
      color: `hsl(${mem.importance * 360}, 70%, 50%)`
    }));
  }, [consciousnessData?.memories]);
  
  const memoryLinks = useMemo(() => {
    if (!processedMemories || processedMemories.length < 2) return [];
    
    // Create links based on similarity (simplified)
    const links = [];
    for (let i = 0; i < processedMemories.length - 1; i++) {
      for (let j = i + 1; j < Math.min(i + 3, processedMemories.length); j++) {
        links.push({
          source: processedMemories[i].id,
          target: processedMemories[j].id,
          value: Math.random(),
          from: processedMemories[i].position,
          to: processedMemories[j].position,
          strength: Math.random()
        });
      }
    }
    return links;
  }, [processedMemories]);
  
  return (
    <div className="enhanced-dashboard">
      <div className="dashboard-header">
        <h2>Consciousness Visualization Suite</h2>
        <div className="view-controls">
          <button
            className={view3D ? 'active' : ''}
            onClick={() => setView3D(true)}
          >
            3D View
          </button>
          <button
            className={!view3D ? 'active' : ''}
            onClick={() => setView3D(false)}
          >
            2D View
          </button>
        </div>
      </div>
      
      <div className="visualization-tabs">
        <button
          className={selectedVisualization === 'core' ? 'active' : ''}
          onClick={() => setSelectedVisualization('core')}
        >
          Consciousness Core
        </button>
        <button
          className={selectedVisualization === 'memory' ? 'active' : ''}
          onClick={() => setSelectedVisualization('memory')}
        >
          Memory Network
        </button>
        <button
          className={selectedVisualization === 'emotion' ? 'active' : ''}
          onClick={() => setSelectedVisualization('emotion')}
        >
          Emotional Landscape
        </button>
        <button
          className={selectedVisualization === 'evolution' ? 'active' : ''}
          onClick={() => setSelectedVisualization('evolution')}
        >
          Evolution Timeline
        </button>
      </div>
      
      <div className="visualization-container">
        {view3D && selectedVisualization === 'core' && (
          <Canvas camera={{ position: [0, 0, 8], fov: 60 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <ConsciousnessCore3D
              coherence={consciousnessData?.coherence || 0.7}
              emotionalState={consciousnessData?.emotionalState || { pleasure: 0.5, arousal: 0.5, dominance: 0.5 }}
              workingMemory={consciousnessData?.workingMemory || 0.5}
            />
            <OrbitControls enableZoom={true} enablePan={true} />
          </Canvas>
        )}
        
        {view3D && selectedVisualization === 'memory' && (
          <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
            <ambientLight intensity={0.3} />
            <pointLight position={[10, 10, 10]} />
            <MemoryConstellation
              memories={processedMemories}
              connections={memoryLinks}
            />
            <OrbitControls enableZoom={true} enablePan={true} />
          </Canvas>
        )}
        
        {!view3D && selectedVisualization === 'memory' && (
          <MemoryForceGraph
            memories={processedMemories}
            links={memoryLinks}
            width={800}
            height={600}
          />
        )}
        
        {selectedVisualization === 'emotion' && (
          <EmotionalLandscape
            emotionHistory={consciousnessData?.emotionHistory || []}
            width={800}
            height={400}
          />
        )}
        
        {selectedVisualization === 'evolution' && (
          <EvolutionTimeline
            events={consciousnessData?.evolutionEvents || []}
            width={800}
            height={200}
          />
        )}
      </div>
      
      <style jsx>{`
        .enhanced-dashboard {
          background: #0a0a0a;
          color: #e0e0e0;
          padding: 20px;
          border-radius: 12px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        
        .dashboard-header h2 {
          margin: 0;
          font-size: 24px;
          font-weight: 600;
        }
        
        .view-controls, .visualization-tabs {
          display: flex;
          gap: 8px;
        }
        
        .view-controls button, .visualization-tabs button {
          padding: 8px 16px;
          background: #1a1a1a;
          border: 1px solid #333;
          color: #e0e0e0;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s;
          font-size: 14px;
        }
        
        .view-controls button:hover, .visualization-tabs button:hover {
          background: #2a2a2a;
          border-color: #444;
        }
        
        .view-controls button.active, .visualization-tabs button.active {
          background: #00a8ff;
          border-color: #00a8ff;
          color: white;
        }
        
        .visualization-tabs {
          margin-bottom: 20px;
          border-bottom: 1px solid #333;
          padding-bottom: 12px;
        }
        
        .visualization-container {
          width: 100%;
          height: 600px;
          background: #050505;
          border: 1px solid #333;
          border-radius: 8px;
          overflow: hidden;
          position: relative;
        }
        
        canvas {
          width: 100% !important;
          height: 100% !important;
        }
        
        svg {
          background: #050505;
        }
        
        .memory-tooltip {
          pointer-events: none;
        }
        
        /* Animations */
        @keyframes pulse {
          0%, 100% { opacity: 0.8; }
          50% { opacity: 1; }
        }
        
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default EnhancedConsciousnessDashboard;