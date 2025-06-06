import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TabGroup = ({ 
  tabs, 
  defaultTab = null, 
  onChange, 
  variant = 'default',
  fullWidth = false 
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);
  
  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    if (onChange) {
      onChange(tabId);
    }
  };
  
  const activeTabData = tabs.find(tab => tab.id === activeTab);
  
  return (
    <div className={`tab-group ${variant} ${fullWidth ? 'full-width' : ''}`}>
      <div className="tab-header">
        <div className="tab-list">
          {tabs.map((tab, index) => (
            <motion.button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''} ${tab.disabled ? 'disabled' : ''}`}
              onClick={() => !tab.disabled && handleTabChange(tab.id)}
              disabled={tab.disabled}
              whileHover={!tab.disabled ? { scale: 1.02 } : {}}
              whileTap={!tab.disabled ? { scale: 0.98 } : {}}
            >
              {tab.icon && <span className="tab-icon">{tab.icon}</span>}
              <span className="tab-label">{tab.label}</span>
              {tab.badge && (
                <span className="tab-badge" style={{ backgroundColor: tab.badgeColor || '#00a8ff' }}>
                  {tab.badge}
                </span>
              )}
              
              {activeTab === tab.id && (
                <motion.div
                  className="tab-indicator"
                  layoutId="activeTab"
                  initial={false}
                  transition={{
                    type: "spring",
                    stiffness: 500,
                    damping: 30
                  }}
                />
              )}
            </motion.button>
          ))}
        </div>
        
        {activeTabData?.actions && (
          <div className="tab-actions">
            {activeTabData.actions}
          </div>
        )}
      </div>
      
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          className="tab-content"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          {activeTabData?.content || activeTabData?.render?.()}
        </motion.div>
      </AnimatePresence>
      
      <style jsx>{`
        .tab-group {
          display: flex;
          flex-direction: column;
          height: 100%;
        }
        
        .tab-group.full-width {
          width: 100%;
        }
        
        .tab-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        
        .tab-list {
          display: flex;
          gap: 4px;
          padding: 4px;
          background: #1a1a1a;
          border-radius: 8px;
          position: relative;
        }
        
        .tab-group.pills .tab-list {
          gap: 8px;
          background: transparent;
        }
        
        .tab-group.underline .tab-list {
          background: transparent;
          border-bottom: 1px solid #333;
          border-radius: 0;
          padding: 0;
          gap: 0;
        }
        
        .tab-button {
          position: relative;
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 16px;
          background: none;
          border: none;
          border-radius: 6px;
          color: #666;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }
        
        .tab-button:hover:not(.disabled) {
          color: #a0a0a0;
        }
        
        .tab-button.active {
          color: #e0e0e0;
        }
        
        .tab-button.disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .tab-group.pills .tab-button {
          background: #252525;
          border: 1px solid #333;
        }
        
        .tab-group.pills .tab-button:hover:not(.disabled) {
          background: #333;
          border-color: #444;
        }
        
        .tab-group.pills .tab-button.active {
          background: #00a8ff;
          border-color: #00a8ff;
          color: white;
        }
        
        .tab-group.underline .tab-button {
          border-radius: 0;
          padding: 12px 20px;
          margin-bottom: -1px;
        }
        
        .tab-icon {
          font-size: 16px;
        }
        
        .tab-label {
          font-size: 14px;
        }
        
        .tab-badge {
          display: flex;
          align-items: center;
          justify-content: center;
          min-width: 18px;
          height: 18px;
          padding: 0 6px;
          background: #00a8ff;
          color: white;
          font-size: 11px;
          font-weight: 600;
          border-radius: 9px;
        }
        
        .tab-indicator {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 100%;
          background: rgba(0, 168, 255, 0.1);
          border-radius: 6px;
          z-index: -1;
        }
        
        .tab-group.pills .tab-indicator {
          display: none;
        }
        
        .tab-group.underline .tab-indicator {
          height: 2px;
          background: #00a8ff;
          border-radius: 0;
          bottom: -1px;
        }
        
        .tab-actions {
          display: flex;
          gap: 8px;
        }
        
        .tab-content {
          flex: 1;
          overflow: auto;
        }
        
        @media (max-width: 640px) {
          .tab-list {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
          }
          
          .tab-list::-webkit-scrollbar {
            display: none;
          }
          
          .tab-button {
            padding: 6px 12px;
            font-size: 13px;
          }
          
          .tab-icon {
            font-size: 14px;
          }
        }
      `}</style>
    </div>
  );
};

export default TabGroup;