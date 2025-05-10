# LearnSmart AI App

A React-based web application built with Vite, featuring Tailwind CSS for styling.

## Prerequisites

- Node.js (version 16 or higher)
- npm or yarn package manager

## Getting Started

1. Clone the repository:
```bash
git clone <repository-url>
cd learnsmart-ai-app
```

2. Install dependencies:
```bash
npm install
# or if using yarn
yarn install
```

3. Start the development server:
```bash
npm run dev
# or if using yarn
yarn dev
```

The application will be available at `http://localhost:5173` by default.

## Available Scripts

- `npm run dev` - Start the development server
- `npm run build` - Build the application for production
- `npm run preview` - Preview the production build locally
- `npm run lint` - Run ESLint to check for code issues

## Tech Stack

- React 19
- Vite 6
- Tailwind CSS 3
- ESLint
- PostCSS
- Autoprefixer

## Project Structure

```
learnsmart-ai-app/
├── public/          # Static assets
├── src/            # Source files
│   ├── assets/     # Images and other assets
│   ├── App.jsx     # Main application component
│   ├── main.jsx    # Application entry point
│   └── index.css   # Global styles and Tailwind imports
├── index.html      # HTML template
├── package.json    # Project dependencies and scripts
├── postcss.config.js # PostCSS configuration
├── tailwind.config.js # Tailwind CSS configuration
└── vite.config.js  # Vite configuration
```

## Development

The project uses:
- Tailwind CSS for styling
- ESLint for code linting
- Vite for fast development and building

## Building for Production

To create a production build:

```bash
npm run build
# or
yarn build
```

The built files will be in the `dist` directory.

## License

[Add your license information here]
