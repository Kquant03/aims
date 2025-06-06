import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const SearchBar = ({ 
  onSearch, 
  onClear,
  placeholder = "Search...",
  suggestions = [],
  showSuggestions = true,
  variant = 'default',
  size = 'medium',
  autoFocus = false
}) => {
  const [value, setValue] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [showSuggestionsList, setShowSuggestionsList] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const inputRef = useRef(null);
  const suggestionsRef = useRef(null);
  
  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);
  
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(event.target)) {
        setShowSuggestionsList(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  const handleInputChange = (e) => {
    const newValue = e.target.value;
    setValue(newValue);
    setSelectedSuggestionIndex(-1);
    
    if (onSearch) {
      onSearch(newValue);
    }
    
    if (showSuggestions && newValue.trim() && suggestions.length > 0) {
      setShowSuggestionsList(true);
    } else {
      setShowSuggestionsList(false);
    }
  };
  
  const handleClear = () => {
    setValue('');
    setShowSuggestionsList(false);
    setSelectedSuggestionIndex(-1);
    
    if (onClear) {
      onClear();
    }
    
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };
  
  const handleSuggestionClick = (suggestion) => {
    setValue(suggestion.text || suggestion);
    setShowSuggestionsList(false);
    
    if (onSearch) {
      onSearch(suggestion.text || suggestion);
    }
  };
  
  const handleKeyDown = (e) => {
    if (!showSuggestionsList || suggestions.length === 0) return;
    
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedSuggestionIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : 0
        );
        break;
        
      case 'ArrowUp':
        e.preventDefault();
        setSelectedSuggestionIndex(prev => 
          prev > 0 ? prev - 1 : suggestions.length - 1
        );
        break;
        
      case 'Enter':
        e.preventDefault();
        if (selectedSuggestionIndex >= 0) {
          handleSuggestionClick(suggestions[selectedSuggestionIndex]);
        }
        break;
        
      case 'Escape':
        setShowSuggestionsList(false);
        setSelectedSuggestionIndex(-1);
        break;
    }
  };
  
  const getSizeClasses = () => {
    const sizes = {
      small: 'search-small',
      medium: 'search-medium',
      large: 'search-large'
    };
    return sizes[size] || sizes.medium;
  };
  
  return (
    <div className={`search-bar ${variant} ${getSizeClasses()} ${isFocused ? 'focused' : ''}`}>
      <div className="search-input-container">
        <div className="search-icon">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="7" cy="7" r="5" stroke="currentColor" strokeWidth="2"/>
            <path d="M11 11L14 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </div>
        
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="search-input"
        />
        
        <AnimatePresence>
          {value && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={handleClear}
              className="clear-button"
              type="button"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M1 1L13 13M1 13L13 1" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </motion.button>
          )}
        </AnimatePresence>
      </div>
      
      <AnimatePresence>
        {showSuggestionsList && suggestions.length > 0 && (
          <motion.div
            ref={suggestionsRef}
            className="suggestions-dropdown"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {suggestions.map((suggestion, index) => {
              const isObject = typeof suggestion === 'object';
              const text = isObject ? suggestion.text : suggestion;
              const meta = isObject ? suggestion.meta : null;
              const icon = isObject ? suggestion.icon : null;
              
              return (
                <motion.div
                  key={index}
                  className={`suggestion-item ${index === selectedSuggestionIndex ? 'selected' : ''}`}
                  onClick={() => handleSuggestionClick(suggestion)}
                  whileHover={{ backgroundColor: 'rgba(0, 168, 255, 0.1)' }}
                >
                  {icon && <span className="suggestion-icon">{icon}</span>}
                  <div className="suggestion-content">
                    <span className="suggestion-text">{text}</span>
                    {meta && <span className="suggestion-meta">{meta}</span>}
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .search-bar {
          position: relative;
          width: 100%;
          max-width: 500px;
        }
        
        .search-input-container {
          position: relative;
          display: flex;
          align-items: center;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          transition: all 0.2s;
        }
        
        .search-bar.focused .search-input-container {
          border-color: #00a8ff;
          box-shadow: 0 0 0 2px rgba(0, 168, 255, 0.2);
        }
        
        .search-bar.minimal .search-input-container {
          background: transparent;
          border-color: #252525;
        }
        
        .search-bar.rounded .search-input-container {
          border-radius: 24px;
        }
        
        .search-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 0 12px;
          color: #666;
        }
        
        .search-input {
          flex: 1;
          background: none;
          border: none;
          color: #e0e0e0;
          outline: none;
          font-size: 14px;
          line-height: 1.5;
        }
        
        .search-small .search-input-container {
          height: 36px;
        }
        
        .search-small .search-input {
          font-size: 13px;
        }
        
        .search-medium .search-input-container {
          height: 44px;
        }
        
        .search-large .search-input-container {
          height: 52px;
        }
        
        .search-large .search-input {
          font-size: 16px;
        }
        
        .search-input::placeholder {
          color: #666;
        }
        
        .clear-button {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 8px;
          margin-right: 4px;
          background: none;
          border: none;
          color: #666;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .clear-button:hover {
          color: #e0e0e0;
        }
        
        .suggestions-dropdown {
          position: absolute;
          top: calc(100% + 8px);
          left: 0;
          right: 0;
          max-height: 300px;
          overflow-y: auto;
          background: #1a1a1a;
          border: 1px solid #333;
          border-radius: 8px;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
          z-index: 100;
        }
        
        .suggestion-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 16px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .suggestion-item:hover {
          background: rgba(0, 168, 255, 0.1);
        }
        
        .suggestion-item.selected {
          background: rgba(0, 168, 255, 0.2);
        }
        
        .suggestion-icon {
          font-size: 16px;
          opacity: 0.8;
        }
        
        .suggestion-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        
        .suggestion-text {
          font-size: 14px;
          color: #e0e0e0;
        }
        
        .suggestion-meta {
          font-size: 12px;
          color: #666;
        }
        
        /* Scrollbar styling */
        .suggestions-dropdown::-webkit-scrollbar {
          width: 6px;
        }
        
        .suggestions-dropdown::-webkit-scrollbar-track {
          background: #252525;
        }
        
        .suggestions-dropdown::-webkit-scrollbar-thumb {
          background: #444;
          border-radius: 3px;
        }
        
        .suggestions-dropdown::-webkit-scrollbar-thumb:hover {
          background: #555;
        }
      `}</style>
    </div>
  );
};

export default SearchBar;