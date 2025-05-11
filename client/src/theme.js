// Central theme configuration file
// Contains all color and style variables for the application

const theme = {
  // Primary app colors
  colors: {
    primary: 'blue-600',
    secondary: 'purple-600',
    success: 'green-500',
    danger: 'red-500',
    warning: 'yellow-500',
    info: 'sky-500',
    gray: 'gray-500',
  },
  
  // Progress bar specific colors
  progressBars: {
    // Linear progress bar
    linear: {
      bg: 'bg-gray-200',          // Background color
      fill: 'bg-blue-600',      // Fill color
      darkBg: 'dark:bg-gray-700', // Dark mode background
      darkFill: 'dark:bg-purple-500' // Dark mode fill
    },
    
    // Circular progress bar
    circular: {
      track: 'text-gray-200',       // Track color
      fill: 'text-purple-600',      // Fill color
      darkTrack: 'dark:text-gray-700', // Dark mode track
      darkFill: 'dark:text-purple-500'   // Dark mode fill
    }
  },
  
  // Button colors
  buttons: {
    primary: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500',
    secondary: 'bg-gray-200 hover:bg-gray-300 focus:ring-gray-400',
    danger: 'bg-red-500 hover:bg-red-600 focus:ring-red-400'
  }
};

// Helper functions to get theme values
export const getColor = (colorName) => {
  return theme.colors[colorName] || theme.colors.primary;
};

export const getProgressBarColors = (type = 'linear') => {
  return theme.progressBars[type];
};

export const getButtonColors = (variant) => {
  return theme.buttons[variant] || theme.buttons.primary;
};

export default theme; 