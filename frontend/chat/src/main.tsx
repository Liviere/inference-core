import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './App';
import { bootstrapTheme } from './lib/theme';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import './index.css';

// Apply the persisted / OS-preferred theme BEFORE React mounts to avoid FOUC.
bootstrapTheme();

ReactDOM.createRoot(document.getElementById('root')!).render(
	<React.StrictMode>
		<App />
	</React.StrictMode>
);
