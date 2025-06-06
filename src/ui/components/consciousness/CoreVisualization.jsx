import React, { useRef, useState, useEffect } from 'react';
import * as THREE from 'three';

const CoreVisualization = ({ coherence = 0.7, emotionalState = {}, workingMemory = [] }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const coreRef = useRef(null);
  const particlesRef = useRef(null);
  const frameIdRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  
  const [isInitialized, setIsInitialized] = useState(false);
  
  useEffect(() => {
    if (!mountRef.current || isInitialized) return;
    
    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    sceneRef.current = scene;
    
    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      60,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 5;
    
    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);
    
    const pointLight1 = new THREE.PointLight(0x00d4ff, 1);
    pointLight1.position.set(5, 5, 5);
    scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0xff00aa, 0.5);
    pointLight2.position.set(-5, -5, 5);
    scene.add(pointLight2);
    
    // Core sphere
    const coreGeometry = new THREE.SphereGeometry(1, 32, 32);
    const coreMaterial = new THREE.MeshPhongMaterial({
      color: new THREE.Color(0x00d4ff),
      emissive: new THREE.Color(0x00d4ff),
      emissiveIntensity: 0.3,
      shininess: 100,
      specular: new THREE.Color(0xffffff),
      transparent: true,
      opacity: 0.9
    });
    const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    scene.add(coreMesh);
    coreRef.current = coreMesh;
    
    // Inner glow
    const glowGeometry = new THREE.SphereGeometry(0.95, 32, 32);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0x00d4ff),
      transparent: true,
      opacity: 0.3
    });
    const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
    coreMesh.add(glowMesh);
    
    // Outer field
    const fieldGeometry = new THREE.SphereGeometry(2.5, 32, 32);
    const fieldMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0x00d4ff),
      transparent: true,
      opacity: 0.05,
      side: THREE.BackSide
    });
    const fieldMesh = new THREE.Mesh(fieldGeometry, fieldMaterial);
    scene.add(fieldMesh);
    
    // Particle system
    const particleCount = 500;
    const particlesGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = 1.5 + Math.random() * 1;
      
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
      
      colors[i * 3] = 0;
      colors[i * 3 + 1] = 0.8 + Math.random() * 0.2;
      colors[i * 3 + 2] = 1;
    }
    
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particlesGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const particlesMaterial = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });
    
    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particles);
    particlesRef.current = particles;
    
    // Mouse movement handler
    const handleMouseMove = (event) => {
      const rect = mountRef.current.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    };
    
    mountRef.current.addEventListener('mousemove', handleMouseMove);
    
    // Animation loop
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      
      const time = Date.now() * 0.001;
      
      // Core pulsing based on coherence
      if (coreRef.current) {
        const scale = 1 + Math.sin(time * 2) * 0.05 * coherence;
        coreRef.current.scale.set(scale, scale, scale);
        
        // Rotation based on emotional arousal
        const arousal = emotionalState.arousal || 0.5;
        coreRef.current.rotation.y += arousal * 0.01;
        coreRef.current.rotation.x = Math.sin(time) * 0.1;
        
        // Color based on emotional state
        const pleasure = emotionalState.pleasure || 0.5;
        const dominance = emotionalState.dominance || 0.5;
        
        const color = new THREE.Color(
          0.2 + pleasure * 0.3,
          0.5 + dominance * 0.3,
          0.8 + (1 - arousal) * 0.2
        );
        coreRef.current.material.color = color;
        coreRef.current.material.emissive = color;
      }
      
      // Particle animation
      if (particlesRef.current) {
        particlesRef.current.rotation.y += 0.001;
        
        const positions = particlesRef.current.geometry.attributes.position;
        const colors = particlesRef.current.geometry.attributes.color;
        
        for (let i = 0; i < particleCount; i++) {
          const i3 = i * 3;
          
          // Orbital motion
          const x = positions.array[i3];
          const z = positions.array[i3 + 2];
          const angle = time * 0.1 + i * 0.01;
          
          positions.array[i3] = x * Math.cos(angle * 0.01) - z * Math.sin(angle * 0.01);
          positions.array[i3 + 2] = x * Math.sin(angle * 0.01) + z * Math.cos(angle * 0.01);
          
          // Color based on working memory activity
          const activity = Math.min(workingMemory.length / 10, 1);
          colors.array[i3 + 1] = 0.8 + activity * 0.2;
        }
        
        positions.needsUpdate = true;
        colors.needsUpdate = true;
      }
      
      // Camera movement based on mouse
      camera.position.x = mouseRef.current.x * 0.5;
      camera.position.y = mouseRef.current.y * 0.5;
      camera.lookAt(scene.position);
      
      // Field opacity based on coherence
      fieldMesh.material.opacity = 0.05 + coherence * 0.1;
      
      renderer.render(scene, camera);
    };
    
    animate();
    setIsInitialized(true);
    
    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Cleanup
    return () => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
      
      if (mountRef.current && rendererRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      
      window.removeEventListener('resize', handleResize);
      mountRef.current?.removeEventListener('mousemove', handleMouseMove);
      
      // Dispose Three.js resources
      scene.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(material => material.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
      
      renderer.dispose();
    };
  }, [isInitialized]);
  
  // Update emotional state colors
  useEffect(() => {
    if (coreRef.current && emotionalState) {
      const pleasure = emotionalState.pleasure || 0.5;
      const arousal = emotionalState.arousal || 0.5;
      const dominance = emotionalState.dominance || 0.5;
      
      const color = new THREE.Color(
        0.2 + pleasure * 0.3,
        0.5 + dominance * 0.3,
        0.8 + (1 - arousal) * 0.2
      );
      
      coreRef.current.material.color = color;
      coreRef.current.material.emissive = color;
      coreRef.current.material.emissiveIntensity = 0.3 + coherence * 0.2;
    }
  }, [emotionalState, coherence]);
  
  return (
    <div className="core-visualization">
      <div ref={mountRef} className="three-container" />
      
      <div className="coherence-overlay">
        <div className="coherence-label">Coherence</div>
        <div className="coherence-value">{(coherence * 100).toFixed(1)}%</div>
        <div className="coherence-bar">
          <div 
            className="coherence-fill"
            style={{ 
              width: `${coherence * 100}%`,
              background: `linear-gradient(90deg, #00d4ff ${coherence * 100}%, transparent ${coherence * 100}%)`
            }}
          />
        </div>
      </div>
      
      <style jsx>{`
        .core-visualization {
          position: relative;
          width: 100%;
          height: 100%;
          min-height: 400px;
          background: #0a0a0a;
          border-radius: 12px;
          overflow: hidden;
        }
        
        .three-container {
          width: 100%;
          height: 100%;
        }
        
        .coherence-overlay {
          position: absolute;
          bottom: 20px;
          left: 20px;
          background: rgba(26, 26, 26, 0.8);
          backdrop-filter: blur(10px);
          padding: 16px;
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .coherence-label {
          font-size: 12px;
          color: #666;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin-bottom: 4px;
        }
        
        .coherence-value {
          font-size: 24px;
          font-weight: 600;
          color: #00d4ff;
          margin-bottom: 8px;
        }
        
        .coherence-bar {
          width: 120px;
          height: 4px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
          overflow: hidden;
        }
        
        .coherence-fill {
          height: 100%;
          transition: width 0.3s ease;
        }
      `}</style>
    </div>
  );
};

export default CoreVisualization;