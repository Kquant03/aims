import React, { Component } from 'react';
import { motion } from 'framer-motion';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0
    };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState(prevState => ({
      error,
      errorInfo,
      errorCount: prevState.errorCount + 1
    }));
    
    // Log to error reporting service if available
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }
  
  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };
  
  handleReload = () => {
    window.location.reload();
  };
  
  render() {
    if (this.state.hasError) {
      const { fallback, showDetails = true } = this.props;
      
      // Use custom fallback if provided
      if (fallback) {
        return fallback(this.state.error, this.handleReset);
      }
      
      // Default error UI
      return (
        <motion.div
          className="error-boundary"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <div className="error-content">
            <div className="error-icon">
              <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                <circle cx="32" cy="32" r="30" stroke="#ff0066" strokeWidth="2" opacity="0.5"/>
                <path d="M32 20V36M32 44V44.01" stroke="#ff0066" strokeWidth="4" strokeLinecap="round"/>
              </svg>
            </div>
            
            <h2 className="error-title">Something went wrong</h2>
            
            <p className="error-message">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            
            {showDetails && this.state.errorInfo && (
              <details className="error-details">
                <summary>Technical details</summary>
                <div className="error-stack">
                  <code>
                    {this.state.error?.stack}
                  </code>
                  <div className="component-stack">
                    <strong>Component Stack:</strong>
                    <pre>{this.state.errorInfo.componentStack}</pre>
                  </div>
                </div>
              </details>
            )}
            
            <div className="error-actions">
              <button onClick={this.handleReset} className="reset-button">
                Try Again
              </button>
              <button onClick={this.handleReload} className="reload-button">
                Reload Page
              </button>
            </div>
            
            {this.state.errorCount > 2 && (
              <p className="error-hint">
                This error has occurred multiple times. Consider reloading the page or contacting support.
              </p>
            )}
          </div>
          
          <style jsx>{`
            .error-boundary {
              display: flex;
              align-items: center;
              justify-content: center;
              min-height: 400px;
              padding: 40px;
              background: #0a0a0a;
              border-radius: 12px;
              border: 1px solid #333;
            }
            
            .error-content {
              max-width: 600px;
              text-align: center;
            }
            
            .error-icon {
              display: flex;
              justify-content: center;
              margin-bottom: 24px;
              animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
              0%, 100% { transform: scale(1); opacity: 0.8; }
              50% { transform: scale(1.05); opacity: 1; }
            }
            
            .error-title {
              margin: 0 0 16px 0;
              font-size: 24px;
              font-weight: 600;
              color: #e0e0e0;
            }
            
            .error-message {
              margin: 0 0 24px 0;
              font-size: 16px;
              color: #888;
              line-height: 1.5;
            }
            
            .error-details {
              margin: 0 0 24px 0;
              padding: 16px;
              background: #1a1a1a;
              border: 1px solid #333;
              border-radius: 8px;
              text-align: left;
            }
            
            .error-details summary {
              cursor: pointer;
              font-size: 14px;
              color: #00a8ff;
              font-weight: 500;
              user-select: none;
            }
            
            .error-details summary:hover {
              color: #33bbff;
            }
            
            .error-stack {
              margin-top: 12px;
              padding: 12px;
              background: #0a0a0a;
              border-radius: 6px;
              overflow-x: auto;
            }
            
            .error-stack code {
              display: block;
              font-family: 'SF Mono', Consolas, monospace;
              font-size: 12px;
              color: #ff0066;
              white-space: pre-wrap;
              word-break: break-word;
            }
            
            .component-stack {
              margin-top: 16px;
              padding-top: 16px;
              border-top: 1px solid #333;
            }
            
            .component-stack strong {
              display: block;
              font-size: 12px;
              color: #888;
              margin-bottom: 8px;
            }
            
            .component-stack pre {
              margin: 0;
              font-family: 'SF Mono', Consolas, monospace;
              font-size: 11px;
              color: #666;
              white-space: pre-wrap;
            }
            
            .error-actions {
              display: flex;
              gap: 12px;
              justify-content: center;
            }
            
            .reset-button, .reload-button {
              padding: 10px 24px;
              border: none;
              border-radius: 8px;
              font-size: 14px;
              font-weight: 600;
              cursor: pointer;
              transition: all 0.2s;
            }
            
            .reset-button {
              background: #00a8ff;
              color: white;
            }
            
            .reset-button:hover {
              background: #0090dd;
              transform: translateY(-1px);
            }
            
            .reload-button {
              background: #252525;
              color: #e0e0e0;
              border: 1px solid #333;
            }
            
            .reload-button:hover {
              background: #333;
              border-color: #444;
            }
            
            .error-hint {
              margin: 16px 0 0 0;
              font-size: 13px;
              color: #ffaa00;
              font-style: italic;
            }
          `}</style>
        </motion.div>
      );
    }
    
    return this.props.children;
  }
}

// Functional wrapper for easier use with hooks
export const withErrorBoundary = (Component, errorBoundaryProps) => {
  const WrappedComponent = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  return WrappedComponent;
};

export default ErrorBoundary;