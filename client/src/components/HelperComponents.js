import React, { useEffect, useState } from 'react';
import { getProgressBarColors } from '../theme';

export const ProgressBar = ({ progress, size = 'h-2.5', colorName = 'primary', customColor = null }) => {
  const progressColors = getProgressBarColors('linear');
  const fillColor = customColor || progressColors.fill;
  const bgColor = progressColors.bg;
  const darkBgColor = progressColors.darkBg;
  
  return (
    <div className={`w-full ${bgColor} ${darkBgColor} rounded-full ${size}`}>
      <div className={`${fillColor} ${size} rounded-full`} style={{ width: `${progress}%` }}></div>
    </div>
  );
};

export const CircularProgressBar = ({ 
  progress, 
  size = 120, 
  strokeWidth = 10, 
  colorName = 'primary',
  customColor = null,
  customTrackColor = null,
  animationDuration = "1s" 
}) => {
  const [animatedProgress, setAnimatedProgress] = useState(0);
  const progressColors = getProgressBarColors('circular');
  
  const fillColor = customColor || progressColors.fill;
  const trackColor = customTrackColor || progressColors.track;
  
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  useEffect(() => {
    // Animate to the target progress
    setAnimatedProgress(progress);
  }, [progress]);

  const offset = circumference - (animatedProgress / 100) * circumference;

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg className="absolute w-full h-full transform -rotate-90">
        <circle
          className={trackColor}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className={fillColor}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          style={{
            transition: `stroke-dashoffset ${animationDuration} ease-out` // Apply CSS transition
          }}
        />
      </svg>
      <span className={`absolute text-xl font-bold ${fillColor}`}>{Math.round(animatedProgress)}%</span>
    </div>
  );
};

export const Card = ({ children, className = "" }) => (
  <div className={`bg-white dark:bg-gray-800 shadow-lg rounded-xl p-6 ${className}`}>
    {children}
  </div>
);

export const Button = ({ children, onClick, variant = 'primary', size = 'md', className = '', icon: Icon, disabled = false }) => {
  const baseStyles = "font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-opacity-50 transition-all duration-150 ease-in-out flex items-center justify-center gap-2";
  const variantStyles = {
    primary: `bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 ${disabled ? 'bg-blue-300 hover:bg-blue-300 cursor-not-allowed' : ''}`,
    secondary: `bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600 focus:ring-gray-400 ${disabled ? 'bg-gray-100 hover:bg-gray-100 cursor-not-allowed' : ''}`,
    danger: `bg-red-500 hover:bg-red-600 text-white focus:ring-red-400 ${disabled ? 'bg-red-300 hover:bg-red-300 cursor-not-allowed' : ''}`,
    ghost: `bg-transparent hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 focus:ring-gray-400 ${disabled ? 'text-gray-400 cursor-not-allowed' : ''}`,
  };
  const sizeStyles = {
    sm: "px-3 py-1.5 text-sm",
    md: "px-4 py-2 text-base",
    lg: "px-6 py-3 text-lg",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
    >
      {Icon && <Icon size={size === 'sm' ? 16 : 20} />}
      {children}
    </button>
  );
};