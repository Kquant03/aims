import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const Tooltip = ({ 
  children, 
  content,
  position = 'top',
  delay = 500,
  maxWidth = 250,
  theme = 'dark',
  arrow = true,
  interactive = false,
  disabled = false
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const triggerRef = useRef(null);
  const tooltipRef = useRef(null);
  const timeoutRef = useRef(null);
  
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);
  
  const calculatePosition = () => {
    if (!triggerRef.current || !tooltipRef.current) return;
    
    const trigger = triggerRef.current.getBoundingClientRect();
    const tooltip = tooltipRef.current.getBoundingClientRect();
    const spacing = arrow ? 12 : 8;
    
    let x = 0;
    let y = 0;
    
    switch (position) {
      case 'top':
        x = trigger.left + trigger.width / 2 - tooltip.width / 2;
        y = trigger.top - tooltip.height - spacing;
        break;
        
      case 'bottom':
        x = trigger.left + trigger.width / 2 - tooltip.width / 2;
        y = trigger.bottom + spacing;
        break;
        
      case 'left':
        x = trigger.left - tooltip.width - spacing;
        y = trigger.top + trigger.height / 2 - tooltip.height / 2;
        break;
        
      case 'right':
        x = trigger.right + spacing;
        y = trigger.top + trigger.height / 2 - tooltip.height / 2;
        break;
        
      case 'top-start':
        x = trigger.left;
        y = trigger.top - tooltip.height - spacing;
        break;
        
      case 'top-end':
        x = trigger.right - tooltip.width;
        y = trigger.top - tooltip.height - spacing;
        break;
        
      case 'bottom-start':
        x = trigger.left;
        y = trigger.bottom + spacing;
        break;
        
      case 'bottom-end':
        x = trigger.right - tooltip.width;
        y = trigger.bottom + spacing;
        break;
    }
    
    // Viewport boundaries check
    const padding = 8;
    x = Math.max(padding, Math.min(x, window.innerWidth - tooltip.width - padding));
    y = Math.max(padding, Math.min(y, window.innerHeight - tooltip.height - padding));
    
    setCoords({ x, y });
  };
  
  const showTooltip = () => {
    if (disabled || !content) return;
    
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, delay);
  };
  
  const hideTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    if (!interactive) {
      setIsVisible(false);
    }
  };
  
  const handleMouseEnter = () => showTooltip();
  const handleMouseLeave = () => hideTooltip();
  const handleFocus = () => showTooltip();
  const handleBlur = () => hideTooltip();
  
  const handleTooltipMouseEnter = () => {
    if (interactive && timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
  };
  
  const handleTooltipMouseLeave = () => {
    if (interactive) {
      setIsVisible(false);
    }
  };
  
  useEffect(() => {
    if (isVisible) {
      calculatePosition();
      window.addEventListener('resize', calculatePosition);
      window.addEventListener('scroll', calculatePosition);
      
      return () => {
        window.removeEventListener('resize', calculatePosition);
        window.removeEventListener('scroll', calculatePosition);
      };
    }
  }, [isVisible]);
  
  const getArrowStyles = () => {
    const arrowSize = 6;
    const styles = {
      width: 0,
      height: 0,
      position: 'absolute',
      borderStyle: 'solid'
    };
    
    const borderColor = theme === 'dark' ? '#1a1a1a' : '#fff';
    
    switch (position) {
      case 'top':
      case 'top-start':
      case 'top-end':
        return {
          ...styles,
          bottom: -arrowSize,
          left: '50%',
          transform: 'translateX(-50%)',
          borderWidth: `${arrowSize}px ${arrowSize}px 0`,
          borderColor: `${borderColor} transparent transparent`
        };
        
      case 'bottom':
      case 'bottom-start':
      case 'bottom-end':
        return {
          ...styles,
          top: -arrowSize,
          left: '50%',
          transform: 'translateX(-50%)',
          borderWidth: `0 ${arrowSize}px ${arrowSize}px`,
          borderColor: `transparent transparent ${borderColor}`
        };
        
      case 'left':
        return {
          ...styles,
          right: -arrowSize,
          top: '50%',
          transform: 'translateY(-50%)',
          borderWidth: `${arrowSize}px 0 ${arrowSize}px ${arrowSize}px`,
          borderColor: `transparent transparent transparent ${borderColor}`
        };
        
      case 'right':
        return {
          ...styles,
          left: -arrowSize,
          top: '50%',
          transform: 'translateY(-50%)',
          borderWidth: `${arrowSize}px ${arrowSize}px ${arrowSize}px 0`,
          borderColor: `transparent ${borderColor} transparent transparent`
        };
        
      default:
        return styles;
    }
  };
  
  return (
    <>
      <span
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onFocus={handleFocus}
        onBlur={handleBlur}
        className="tooltip-trigger"
      >
        {children}
      </span>
      
      <AnimatePresence>
        {isVisible && content && (
          <motion.div
            ref={tooltipRef}
            className={`tooltip ${theme}`}
            style={{
              position: 'fixed',
              left: coords.x,
              top: coords.y,
              maxWidth,
              zIndex: 9999
            }}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.15 }}
            onMouseEnter={handleTooltipMouseEnter}
            onMouseLeave={handleTooltipMouseLeave}
          >
            <div className="tooltip-content">
              {content}
            </div>
            {arrow && <div className="tooltip-arrow" style={getArrowStyles()} />}
          </motion.div>
        )}
      </AnimatePresence>
      
      <style jsx>{`
        .tooltip-trigger {
          display: inline-block;
        }
        
        .tooltip {
          pointer-events: ${interactive ? 'auto' : 'none'};
          filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.3));
        }
        
        .tooltip-content {
          padding: 8px 12px;
          border-radius: 6px;
          font-size: 13px;
          line-height: 1.5;
          word-wrap: break-word;
        }
        
        .tooltip.dark .tooltip-content {
          background: #1a1a1a;
          color: #e0e0e0;
          border: 1px solid #333;
        }
        
        .tooltip.light .tooltip-content {
          background: #fff;
          color: #1a1a1a;
          border: 1px solid #e0e0e0;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .tooltip.info .tooltip-content {
          background: rgba(0, 168, 255, 0.1);
          color: #00a8ff;
          border: 1px solid rgba(0, 168, 255, 0.3);
        }
        
        .tooltip.success .tooltip-content {
          background: rgba(0, 255, 136, 0.1);
          color: #00ff88;
          border: 1px solid rgba(0, 255, 136, 0.3);
        }
        
        .tooltip.warning .tooltip-content {
          background: rgba(255, 170, 0, 0.1);
          color: #ffaa00;
          border: 1px solid rgba(255, 170, 0, 0.3);
        }
        
        .tooltip.error .tooltip-content {
          background: rgba(255, 0, 102, 0.1);
          color: #ff0066;
          border: 1px solid rgba(255, 0, 102, 0.3);
        }
        
        .tooltip-arrow {
          pointer-events: none;
        }
      `}</style>
    </>
  );
};

// Compound component for rich tooltip content
export const TooltipContent = ({ title, description, footer }) => (
  <div className="tooltip-rich-content">
    {title && <div className="tooltip-title">{title}</div>}
    {description && <div className="tooltip-description">{description}</div>}
    {footer && <div className="tooltip-footer">{footer}</div>}
    
    <style jsx>{`
      .tooltip-rich-content {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }
      
      .tooltip-title {
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 4px;
      }
      
      .tooltip-description {
        font-size: 13px;
        opacity: 0.9;
        line-height: 1.5;
      }
      
      .tooltip-footer {
        font-size: 12px;
        opacity: 0.7;
        padding-top: 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
      }
    `}</style>
  </div>
);

export default Tooltip;