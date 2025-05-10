// src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import AppWrapper from './App.jsx'; // Correctly imports the default export
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AppWrapper />
  </React.StrictMode>
);
