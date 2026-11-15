import { createContext } from 'react';

// Kept apart from the provider component so React fast refresh works on both files.
export const InteractionContext = createContext(null);
