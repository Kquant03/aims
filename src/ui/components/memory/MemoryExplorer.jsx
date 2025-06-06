// src/ui/components/memory/MemoryExplorer.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MemoryGraph from './MemoryGraph';
import MemoryTimeline from './MemoryTimeline';
import MemorySearch from './MemorySearch';
import MemoryDetail from './MemoryDetail';
import './MemoryExplorer.css';

const MemoryExplorer = ({ sessionId }) => {
  const [memories, setMemories] = useState([]);
  const [selectedMemory, setSelectedMemory] = useState(null);
  const [viewMode, setViewMode] = useState('graph');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [memoryStats, setMemoryStats] = useState(null);
  
  // Fetch memories
  useEffect(() => {
    fetchMemories();
    fetchMemoryStats();
  }, [sessionId]);
  
  const fetchMemories = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/memories?limit=100`, {
        credentials: 'include'
      });
      const data = await response.json();
      setMemories(data);
    } catch (error) {
      console.error('Failed to fetch memories:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const fetchMemoryStats = async () => {
    try {
      const response = await fetch('/api/memory/stats', {
        credentials: 'include'
      });
      const data = await response.json();
      setMemoryStats(data);
    } catch (error) {
      console.error('Failed to fetch memory stats:', error);
    }
  };
  
  const handleSearch = async (query) => {
    setSearchQuery(query);
    if (!query) {
      fetchMemories();
      return;
    }
    
    setIsLoading(true);
    try {
      const response = await fetch(`/api/memories?search=${encodeURIComponent(query)}`, {
        credentials: 'include'
      });
      const data = await response.json();
      setMemories(data);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Filter memories based on search
  const filteredMemories = useMemo(() => {
    if (!searchQuery) return memories;
    
    const query = searchQuery.toLowerCase();
    return memories.filter(memory => 
      memory.content.toLowerCase().includes(query) ||
      memory.type?.toLowerCase().includes(query)
    );
  }, [memories, searchQuery]);
  
  // Generate links for memory graph
  const memoryLinks = useMemo(() => {
    const links = [];
    // Simple link generation - in reality, these would come from the backend
    for (let i = 0; i < memories.length - 1; i++) {
      for (let j = i + 1; j < Math.min(i + 5, memories.length); j++) {
        if (Math.random() > 0.7) { // Randomly connect some memories
          links.push({
            source: memories[i].id,
            target: memories[j].id,
            value: Math.random()
          });
        }
      }
    }
    return links;
  }, [memories]);
  
  return (
    <div className="memory-explorer">
      {/* Header */}
      <div className="explorer-header">
        <h2>Memory Explorer</h2>
        
        {memoryStats && (
          <div className="memory-stats">
            <div className="stat">
              <span className="stat-value">{memoryStats.total_memories || 0}</span>
              <span className="stat-label">Total Memories</span>
            </div>
            <div className="stat">
              <span className="stat-value">{memoryStats.semantic_count || 0}</span>
              <span className="stat-label">Concepts</span>
            </div>
            <div className="stat">
              <span className="stat-value">{memoryStats.avg_importance?.toFixed(2) || '0.00'}</span>
              <span className="stat-label">Avg Importance</span>
            </div>
          </div>
        )}
      </div>
      
      {/* Controls */}
      <div className="explorer-controls">
        <MemorySearch 
          onSearch={handleSearch}
          placeholder="Search memories..."
        />
        
        <div className="view-switcher">
          <button
            className={viewMode === 'graph' ? 'active' : ''}
            onClick={() => setViewMode('graph')}
          >
            <span className="icon">🕸️</span> Graph
          </button>
          <button
            className={viewMode === 'timeline' ? 'active' : ''}
            onClick={() => setViewMode('timeline')}
          >
            <span className="icon">📅</span> Timeline
          </button>
          <button
            className={viewMode === 'semantic' ? 'active' : ''}
            onClick={() => setViewMode('semantic')}
          >
            <span className="icon">🧩</span> Concepts
          </button>
        </div>
      </div>
      
      {/* Main Content */}
      <div className="explorer-content">
        {isLoading ? (
          <div className="loading-state">
            <div className="memory-loader">
              <div className="loader-ring" />
              <p>Loading memories...</p>
            </div>
          </div>
        ) : (
          <>
            <AnimatePresence mode="wait">
              <motion.div
                key={viewMode}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className="memory-view"
              >
                {viewMode === 'graph' && (
                  <MemoryGraph
                    memories={filteredMemories}
                    links={memoryLinks}
                    onMemorySelect={setSelectedMemory}
                    selectedId={selectedMemory?.id}
                  />
                )}
                
                {viewMode === 'timeline' && (
                  <MemoryTimeline
                    memories={filteredMemories}
                    onMemorySelect={setSelectedMemory}
                    selectedId={selectedMemory?.id}
                  />
                )}
                
                {viewMode === 'semantic' && (
                  <SemanticKnowledgeView
                    memories={filteredMemories}
                    onConceptSelect={(concept) => {
                      // Handle concept selection
                      handleSearch(concept);
                    }}
                  />
                )}
              </motion.div>
            </AnimatePresence>
            
            {/* Memory Detail Panel */}
            <AnimatePresence>
              {selectedMemory && (
                <MemoryDetail
                  memory={selectedMemory}
                  onClose={() => setSelectedMemory(null)}
                />
              )}
            </AnimatePresence>
          </>
        )}
      </div>
    </div>
  );
};

// Semantic Knowledge View Component
const SemanticKnowledgeView = ({ memories, onConceptSelect }) => {
  // Extract concepts from memories
  const concepts = useMemo(() => {
    const conceptMap = new Map();
    
    memories.forEach(memory => {
      // Extract concepts (simplified - would come from backend)
      const words = memory.content.split(/\s+/);
      words.forEach(word => {
        if (word.length > 4) {
          const concept = word.toLowerCase();
          conceptMap.set(concept, (conceptMap.get(concept) || 0) + 1);
        }
      });
    });
    
    return Array.from(conceptMap.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 50)
      .map(([concept, count]) => ({ concept, count }));
  }, [memories]);
  
  return (
    <div className="semantic-view">
      <div className="concept-cloud">
        {concepts.map(({ concept, count }) => (
          <button
            key={concept}
            className="concept-tag"
            style={{
              fontSize: `${Math.max(12, Math.min(24, count * 2))}px`,
              opacity: Math.max(0.6, Math.min(1, count / 10))
            }}
            onClick={() => onConceptSelect(concept)}
          >
            {concept}
          </button>
        ))}
      </div>
    </div>
  );
};

export default MemoryExplorer;