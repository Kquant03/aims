import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Trail, MeshDistortMaterial } from '@react-three/drei';
import { motion } from 'framer-motion';
import './EmotionalLandscape.css';

// Emotional state particle system
const EmotionalParticles = ({ currentState }) => {
  const particlesRef = useRef();
  const particleCount = 500;
  
  const positions = React.useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = 3 + Math.random();
      
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
    }
    return pos;
  }, []);
  
  useFrame((state) => {
    if (particlesRef.current) {
      const time = state.clock.elapsedTime;
      const positions = particlesRef.current.geometry.attributes.position;
      
      for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        const x = positions.array[i3];
        const y = positions.array[i3 + 1];
        const z = positions.array[i3 + 2];
        
        // Swirl based on arousal
        const swirl = currentState.arousal * 0.02;
        positions.array[i3] = x * Math.cos(swirl) - z * Math.sin(swirl);
        positions.array[i3 + 2] = x * Math.sin(swirl) + z * Math.cos(swirl);
        
        // Pulse based on pleasure
        const pulse = 1 + Math.sin(time * 2 + i * 0.1) * 0.1 * currentState.pleasure;
        positions.array[i3] *= pulse;
        positions.array[i3 + 1] *= pulse;
        positions.array[i3 + 2] *= pulse;
      }
      
      positions.needsUpdate = true;
    }
  });
  
  const color = new THREE.Color(
    currentState.pleasure,
    1 - currentState.arousal,
    currentState.dominance
  );
  
  return (
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
        color={color}
        transparent
        opacity={0.6}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

// Current emotional state sphere
const EmotionalCore = ({ currentState, history }) => {
  const meshRef = useRef();
  const trailRef = useRef();
  
  // Convert PAD to 3D position
  const position = [
    (currentState.pleasure - 0.5) * 4,
    (currentState.arousal - 0.5) * 4,
    (currentState.dominance - 0.5) * 4
  ];
  
  useFrame((state) => {
    if (meshRef.current) {
      // Gentle rotation
      meshRef.current.rotation.x += 0.001;
      meshRef.current.rotation.y += 0.002;
      
      // Pulsing based on emotional intensity
      const intensity = Math.sqrt(
        Math.pow(currentState.pleasure - 0.5, 2) +
        Math.pow(currentState.arousal - 0.5, 2) +
        Math.pow(currentState.dominance - 0.5, 2)
      ) / Math.sqrt(0.75);
      
      const scale = 0.5 + Math.sin(state.clock.elapsedTime * 3) * 0.1 * intensity;
      meshRef.current.scale.setScalar(scale);
    }
  });
  
  const color = new THREE.Color(
    currentState.pleasure,
    1 - currentState.arousal,
    currentState.dominance
  );
  
  return (
    <group position={position}>
      <Trail
        width={2}
        length={20}
        color={color}
        attenuation={(t) => t * t}
      >
        <mesh ref={meshRef}>
          <sphereGeometry args={[0.3, 32, 32]} />
          <MeshDistortMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.5}
            roughness={0.3}
            metalness={0.7}
            distort={0.3}
            speed={2}
          />
        </mesh>
      </Trail>
      
      {/* Emotional field */}
      <mesh>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.1}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
};

// Axis labels and grid
const EmotionalAxes = () => {
  return (
    <>
      {/* Pleasure axis (X) - Red */}
      <Line
        points={[[-4, 0, 0], [4, 0, 0]]}
        color="#ff0066"
        lineWidth={2}
      />
      <Text
        position={[4.5, 0, 0]}
        fontSize={0.3}
        color="#ff0066"
      >
        Pleasure →
      </Text>
      <Text
        position={[-4.5, 0, 0]}
        fontSize={0.3}
        color="#ff0066"
      >
        ← Displeasure
      </Text>
      
      {/* Arousal axis (Y) - Green */}
      <Line
        points={[[0, -4, 0], [0, 4, 0]]}
        color="#00ff88"
        lineWidth={2}
      />
      <Text
        position={[0, 4.5, 0]}
        fontSize={0.3}
        color="#00ff88"
      >
        High Arousal
      </Text>
      <Text
        position={[0, -4.5, 0]}
        fontSize={0.3}
        color="#00ff88"
      >
        Low Arousal
      </Text>
      
      {/* Dominance axis (Z) - Blue */}
      <Line
        points={[[0, 0, -4], [0, 0, 4]]}
        color="#00a8ff"
        lineWidth={2}
      />
      <Text
        position={[0, 0, 4.5]}
        fontSize={0.3}
        color="#00a8ff"
      >
        Dominance
      </Text>
      <Text
        position={[0, 0, -4.5]}
        fontSize={0.3}
        color="#00a8ff"
      >
        Submission
      </Text>
      
      {/* Grid planes */}
      <gridHelper args={[8, 8, '#333333', '#222222']} rotation={[Math.PI / 2, 0, 0]} />
      <gridHelper args={[8, 8, '#333333', '#222222']} rotation={[0, 0, 0]} position={[0, -4, 0]} />
      <gridHelper args={[8, 8, '#333333', '#222222']} rotation={[0, 0, Math.PI / 2]} position={[-4, 0, 0]} />
    </>
  );
};

// Emotion labels in 3D space
const EmotionLabels = ({ emotions }) => {
  const emotionPositions = {
    joy: [3, 3, 3],
    excitement: [2.5, 3.5, 2],
    contentment: [3, -2, 2],
    serenity: [3.5, -3, 1],
    anger: [-3, 3, 3],
    fear: [-3, 3, -3],
    anxiety: [-2, 3, -2],
    sadness: [-3, -2, -2],
    boredom: [0, -3, 0],
    neutral: [0, 0, 0],
    curiosity: [1, 1, 1],
    surprise: [0, 3, -1],
    disgust: [-3, 1, 1],
    contempt: [-2, 0, 3],
    pride: [2, 1, 3.5],
    shame: [-3, 0, -3],
    guilt: [-2, 1, -2]
  };
  
  return (
    <>
      {Object.entries(emotionPositions).map(([emotion, position]) => (
        <group key={emotion} position={position}>
          <Text
            fontSize={0.2}
            color="#666666"
            anchorX="center"
            anchorY="middle"
          >
            {emotion}
          </Text>
          <mesh>
            <sphereGeometry args={[0.1, 16, 16]} />
            <meshBasicMaterial color="#333333" transparent opacity={0.5} />
          </mesh>
        </group>
      ))}
    </>
  );
};

// History trail visualization
const EmotionalTrail = ({ history }) => {
  const points = React.useMemo(() => {
    return history.slice(-50).map(state => [
      (state.pleasure - 0.5) * 4,
      (state.arousal - 0.5) * 4,
      (state.dominance - 0.5) * 4
    ]);
  }, [history]);
  
  if (points.length < 2) return null;
  
  return (
    <Line
      points={points}
      color="#ffffff"
      lineWidth={1}
      opacity={0.3}
      transparent
    />
  );
};

const EmotionalLandscape = ({ currentState, history = [] }) => {
  const [showLabels, setShowLabels] = useState(true);
  const [showTrail, setShowTrail] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);
  
  // Default state if not provided
  const state = currentState || {
    pleasure: 0.5,
    arousal: 0.5,
    dominance: 0.5,
    label: 'neutral'
  };
  
  // Calculate emotional metrics
  const intensity = Math.sqrt(
    Math.pow(state.pleasure - 0.5, 2) +
    Math.pow(state.arousal - 0.5, 2) +
    Math.pow(state.dominance - 0.5, 2)
  ) / Math.sqrt(0.75);
  
  return (
    <div className="emotional-landscape-container">
      <div className="landscape-controls">
        <button
          className={showLabels ? 'active' : ''}
          onClick={() => setShowLabels(!showLabels)}
        >
          <span className="icon">🏷️</span> Labels
        </button>
        <button
          className={showTrail ? 'active' : ''}
          onClick={() => setShowTrail(!showTrail)}
        >
          <span className="icon">〰️</span> Trail
        </button>
        <button
          className={autoRotate ? 'active' : ''}
          onClick={() => setAutoRotate(!autoRotate)}
        >
          <span className="icon">🔄</span> Auto-rotate
        </button>
      </div>
      
      <div className="emotional-info">
        <div className="current-emotion">
          <h3>{state.label || 'Unknown'}</h3>
          <div className="emotion-metrics">
            <div className="metric">
              <span className="label">Pleasure</span>
              <div className="bar">
                <div 
                  className="fill pleasure"
                  style={{ width: `${state.pleasure * 100}%` }}
                />
              </div>
              <span className="value">{(state.pleasure * 100).toFixed(0)}%</span>
            </div>
            <div className="metric">
              <span className="label">Arousal</span>
              <div className="bar">
                <div 
                  className="fill arousal"
                  style={{ width: `${state.arousal * 100}%` }}
                />
              </div>
              <span className="value">{(state.arousal * 100).toFixed(0)}%</span>
            </div>
            <div className="metric">
              <span className="label">Dominance</span>
              <div className="bar">
                <div 
                  className="fill dominance"
                  style={{ width: `${state.dominance * 100}%` }}
                />
              </div>
              <span className="value">{(state.dominance * 100).toFixed(0)}%</span>
            </div>
          </div>
          <div className="intensity-indicator">
            <span>Intensity: </span>
            <strong style={{ color: `hsl(${intensity * 120}, 70%, 50%)` }}>
              {(intensity * 100).toFixed(0)}%
            </strong>
          </div>
        </div>
      </div>
      
      <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.5} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} />
        
        <EmotionalAxes />
        {showLabels && <EmotionLabels />}
        <EmotionalCore currentState={state} history={history} />
        <EmotionalParticles currentState={state} />
        {showTrail && history.length > 0 && <EmotionalTrail history={history} />}
        
        <OrbitControls
          enableZoom={true}
          enablePan={true}
          autoRotate={autoRotate}
          autoRotateSpeed={0.5}
        />
        
        {/* Background sphere */}
        <mesh>
          <sphereGeometry args={[50, 32, 32]} />
          <meshBasicMaterial
            color="#0a0a0a"
            side={THREE.BackSide}
          />
        </mesh>
      </Canvas>
      
      <style jsx>{`
        .emotional-landscape-container {
          position: relative;
          width: 100%;
          height: 100%;
          background: #0a0a0a;
          border-radius: 8px;
          overflow: hidden;
        }
        
        .landscape-controls {
          position: absolute;
          top: 16px;
          right: 16px;
          display: flex;
          gap: 8px;
          z-index: 10;
        }
        
        .landscape-controls button {
          background: rgba(26, 26, 26, 0.8);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          padding: 8px 12px;
          color: #666;
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .landscape-controls button:hover {
          color: #e0e0e0;
          background: rgba(255, 255, 255, 0.05);
        }
        
        .landscape-controls button.active {
          color: #00ff88;
          border-color: rgba(0, 255, 136, 0.3);
        }
        
        .emotional-info {
          position: absolute;
          top: 16px;
          left: 16px;
          background: rgba(26, 26, 26, 0.9);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 16px;
          max-width: 250px;
          z-index: 10;
        }
        
        .current-emotion h3 {
          margin: 0 0 12px 0;
          color: #e0e0e0;
          font-size: 18px;
          text-transform: capitalize;
        }
        
        .emotion-metrics {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 12px;
        }
        
        .metric {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .metric .label {
          font-size: 11px;
          color: #888;
          width: 60px;
        }
        
        .bar {
          flex: 1;
          height: 4px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
          overflow: hidden;
        }
        
        .fill {
          height: 100%;
          transition: width 0.3s ease;
        }
        
        .fill.pleasure {
          background: #ff0066;
        }
        
        .fill.arousal {
          background: #00ff88;
        }
        
        .fill.dominance {
          background: #00a8ff;
        }
        
        .metric .value {
          font-size: 11px;
          color: #e0e0e0;
          width: 35px;
          text-align: right;
        }
        
        .intensity-indicator {
          font-size: 12px;
          color: #888;
        }
        
        .intensity-indicator strong {
          font-weight: 600;
        }
      `}</style>
    </div>
  );
};

export default EmotionalLandscape;