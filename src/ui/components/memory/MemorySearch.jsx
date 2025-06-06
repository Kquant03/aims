import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const MemorySearch = ({ onSearch, placeholder = "Search memories...", initialValue = "" }) => {
  const [searchValue, setSearchValue] = useState(initialValue);
  const [isSearching, setIsSearching] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    type: 'all',
    timeRange: 'all',
    importance: 'all',
    emotion: 'all'
  });
  
  const searchRef = useRef(null);
  const debounceTimer = useRef(null);
  
  // Debounced search
  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }
    
    if (searchValue.trim()) {
      setIsSearching(true);
      debounceTimer.current = setTimeout(() => {
        performSearch();
      }, 300);
    } else {
      setIsSearching(false);
      onSearch('');
    }
    
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [searchValue, filters]);
  
  const performSearch = () => {
    const searchQuery = buildSearchQuery();
    onSearch(searchQuery);
    setIsSearching(false);
  };
  
  const buildSearchQuery = () => {
    let query = searchValue;
    
    // Add filter parameters
    if (filters.type !== 'all') {
      query += ` type:${filters.type}`;
    }
    if (filters.timeRange !== 'all') {
      query += ` time:${filters.timeRange}`;
    }
    if (filters.importance !== 'all') {
      query += ` importance:${filters.importance}`;
    }
    if (filters.emotion !== 'all') {
      query += ` emotion:${filters.emotion}`;
    }
    
    return query.trim();
  };
  
  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
  };
  
  const clearSearch = () => {
    setSearchValue('');
    setFilters({
      type: 'all',
      timeRange: 'all',
      importance: 'all',
      emotion: 'all'
    });
    setSuggestions([]);
  };
  
  const activeFilterCount = Object.values(filters).filter(v => v !== 'all').length;
  
  return (
    <div className="memory-search">
      <div className="search-container">
        <div className="search-input-wrapper">
          <div className="search-icon">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <circle cx="7" cy="7" r="5" stroke="currentColor" strokeWidth="2"/>
              <path d="M11 11L14 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          
          <input
            ref={searchRef}
            type="text"
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            placeholder={placeholder}
            className="search-input"
          />
          
          {isSearching && (
            <div className="search-loading">
              <div className="spinner" />
            </div>
          )}
          
          {searchValue && (
            <button onClick={clearSearch} className="clear-button">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M1 1L13 13M1 13L13 1" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
          )}
          
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`filter-button ${activeFilterCount > 0 ? 'has-filters' : ''}`}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 4H14M4 8H12M6 12H10" stroke="currentColor" strokeWidth="2"/>
            </svg>
            {activeFilterCount > 0 && (
              <span className="filter-count">{activeFilterCount}</span>
            )}
          </button>
        </div>
      </div>
      
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="filter-panel"
          >
            <div className="filter-section">
              <label>Type</label>
              <select
                value={filters.type}
                onChange={(e) => handleFilterChange('type', e.target.value)}
                className="filter-select"
              >
                <option value="all">All Types</option>
                <option value="conversation">Conversation</option>
                <option value="insight">Insight</option>
                <option value="emotion">Emotional</option>
                <option value="goal">Goal</option>
                <option value="learning">Learning</option>
              </select>
            </div>
            
            <div className="filter-section">
              <label>Time Range</label>
              <select
                value={filters.timeRange}
                onChange={(e) => handleFilterChange('timeRange', e.target.value)}
                className="filter-select"
              >
                <option value="all">All Time</option>
                <option value="today">Today</option>
                <option value="week">This Week</option>
                <option value="month">This Month</option>
                <option value="year">This Year</option>
              </select>
            </div>
            
            <div className="filter-section">
              <label>Importance</label>
              <select
                value={filters.importance}
                onChange={(e) => handleFilterChange('importance', e.target.value)}
                className="filter-select"
              >
                <option value="all">All</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
            
            <div className="filter-section">
              <label>Emotional Context</label>
              <select
                value={filters.emotion}
                onChange={(e) => handleFilterChange('emotion', e.target.value)}
                className="filter-select"
              >
                <option value="all">All Emotions</option>
                <option value="joy">Joy</option>
                <option value="curiosity">Curiosity</option>
                <option value="calm">Calm</option>
                <option value="excitement">Excitement</option>
                <option value="concern">Concern</option>
              </select>
            </div>
            
            {activeFilterCount > 0 && (
              <button onClick={clearSearch} className="clear-filters-button">
                Clear All Filters
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .memory-search {
          width: 100%;
          max-width: 600px;
        }
        
        .search-container {
          position: relative;
        }
        
        .search-input-wrapper {
          display: flex;
          align-items: center;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          transition: all 0.2s;
        }
        
        .search-input-wrapper:focus-within {
          border-color: #00a8ff;
          box-shadow: 0 0 0 2px rgba(0, 168, 255, 0.2);
        }
        
        .search-icon {
          padding: 0 12px;
          color: #666;
        }
        
        .search-input {
          flex: 1;
          background: none;
          border: none;
          color: #e0e0e0;
          font-size: 14px;
          padding: 12px 0;
          outline: none;
        }
        
        .search-input::placeholder {
          color: #666;
        }
        
        .search-loading {
          padding: 0 12px;
        }
        
        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid #333;
          border-top-color: #00a8ff;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .clear-button, .filter-button {
          padding: 8px 12px;
          background: none;
          border: none;
          color: #666;
          cursor: pointer;
          transition: color 0.2s;
        }
        
        .clear-button:hover, .filter-button:hover {
          color: #e0e0e0;
        }
        
        .filter-button {
          position: relative;
          border-left: 1px solid #333;
        }
        
        .filter-button.has-filters {
          color: #00a8ff;
        }
        
        .filter-count {
          position: absolute;
          top: 2px;
          right: 2px;
          width: 14px;
          height: 14px;
          background: #00a8ff;
          color: white;
          font-size: 10px;
          font-weight: bold;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .filter-panel {
          margin-top: 12px;
          padding: 16px;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
          gap: 12px;
        }
        
        .filter-section {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        
        .filter-section label {
          font-size: 12px;
          color: #888;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .filter-select {
          padding: 8px;
          background: #0a0a0a;
          border: 1px solid #333;
          border-radius: 4px;
          color: #e0e0e0;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .filter-select:hover {
          border-color: #444;
        }
        
        .filter-select:focus {
          outline: none;
          border-color: #00a8ff;
        }
        
        .clear-filters-button {
          grid-column: 1 / -1;
          padding: 8px 16px;
          background: #333;
          border: none;
          border-radius: 4px;
          color: #e0e0e0;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .clear-filters-button:hover {
          background: #444;
        }
      `}</style>
    </div>
  );
};

export default MemorySearch;