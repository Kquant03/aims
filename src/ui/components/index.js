// src/ui/components/index.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import '../static/css/main.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);