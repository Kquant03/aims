import React from 'react';
import { motion } from 'framer-motion';

const LoadingSpinner = ({ 
  size = 'medium', 
  variant = 'default',
  message = '',
  fullScreen = false,
  delay = 0
}) => {
  const sizes = {
    small: 24,
    medium: 40,
    large: 60,
    xlarge: 80
  };
  
  const spinnerSize = sizes[size] || sizes.medium;
  
  const variants = {
    default: {
      spinner: (
        <motion.div
          className="spinner-default"
          animate={{ rotate: 360 }}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "linear"
          }}
          style={{
            width: spinnerSize,
            height: spinnerSize,
            border: `3px solid #333`,
            borderTopColor: '#00a8ff',
            borderRadius: '50%'
          }}
        />
      )
    },
    
    pulse: {
      spinner: (
        <motion.div
          className="spinner-pulse"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [1, 0.5, 1]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          style={{
            width: spinnerSize,
            height: spinnerSize,
            background: 'radial-gradient(circle, #00a8ff 0%, transparent 70%)',
            borderRadius: '50%'
          }}
        />
      )
    },
    
    dots: {
      spinner: (
        <div className="spinner-dots" style={{ display: 'flex', gap: spinnerSize * 0.2 }}>
          {[0, 1, 2].map(i => (
            <motion.div
              key={i}
              animate={{
                y: [-spinnerSize * 0.3, 0, -spinnerSize * 0.3]
              }}
              transition={{
                duration: 0.6,
                repeat: Infinity,
                delay: i * 0.2
              }}
              style={{
                width: spinnerSize * 0.25,
                height: spinnerSize * 0.25,
                backgroundColor: '#00a8ff',
                borderRadius: '50%'
              }}
            />
          ))}
        </div>
      )
    },
    
    consciousness: {
      spinner: (
        <div className="spinner-consciousness" style={{ position: 'relative', width: spinnerSize, height: spinnerSize }}>
          <motion.div
            animate={{ rotate: 360 }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "linear"
            }}
            style={{
              position: 'absolute',
              width: '100%',
              height: '100%',
              border: `2px solid #00a8ff`,
              borderRadius: '50%',
              opacity: 0.3
            }}
          />
          <motion.div
            animate={{ rotate: -360 }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear"
            }}
            style={{
              position: 'absolute',
              width: '70%',
              height: '70%',
              top: '15%',
              left: '15%',
              border: `2px solid #ff00aa`,
              borderRadius: '50%',
              opacity: 0.5
            }}
          />
          <motion.div
            animate={{
              scale: [0.8, 1.2, 0.8],
              opacity: [0.5, 1, 0.5]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            style={{
              position: 'absolute',
              width: '30%',
              height: '30%',
              top: '35%',
              left: '35%',
              backgroundColor: '#00ff88',
              borderRadius: '50%'
            }}
          />
        </div>
      )
    },
    
    thinking: {
      spinner: (
        <div className="spinner-thinking" style={{ position: 'relative', width: spinnerSize, height: spinnerSize }}>
          {[0, 1, 2].map(i => (
            <motion.div
              key={i}
              animate={{
                scale: [0, 1, 0],
                opacity: [0, 1, 0]
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                delay: i * 0.5
              }}
              style={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                border: `2px solid #00a8ff`,
                borderRadius: '50%'
              }}
            />
          ))}
        </div>
      )
    }
  };
  
  const content = (
    <motion.div
      className={`loading-spinner ${size} ${variant}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay }}
    >
      <div className="spinner-container">
        {variants[variant]?.spinner || variants.default.spinner}
      </div>
      
      {message && (
        <motion.p
          className="loading-message"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: delay + 0.2 }}
        >
          {message}
        </motion.p>
      )}
    </motion.div>
  );
  
  if (fullScreen) {
    return (
      <div className="loading-fullscreen">
        {content}
      </div>
    );
  }
  
  return (
    <>
      {content}
      <style jsx>{`
        .loading-spinner {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
        }
        
        .spinner-container {
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .loading-message {
          margin: 0;
          font-size: 14px;
          color: #888;
          text-align: center;
          animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        
        .loading-fullscreen {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(10, 10, 10, 0.9);
          backdrop-filter: blur(10px);
          z-index: 9999;
        }
        
        /* Size variations */
        .loading-spinner.small .loading-message {
          font-size: 12px;
        }
        
        .loading-spinner.large .loading-message {
          font-size: 16px;
        }
        
        .loading-spinner.xlarge .loading-message {
          font-size: 18px;
        }
      `}</style>
    </>
  );
};

export default LoadingSpinner;