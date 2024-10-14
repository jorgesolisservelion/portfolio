// app/page.tsx

'use client';

import { useState } from 'react';
import StoryCrafter from './components/StoryCrafter';
import FoodRecipes from './components/FoodRecipes';

export default function Home() {
  const [selectedApp, setSelectedApp] = useState<'story' | 'recipe' | null>(null);

  return (
    <div className="flex flex-col items-center w-full max-w-md py-24 mx-auto">
      <h1 className="text-4xl font-bold mb-8 text-center text-blue-600 font-serif">
        AI-Powered Apps
      </h1>
      
      {!selectedApp ? (
        <div className="flex flex-col items-center">
          <button
            onClick={() => setSelectedApp('story')}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mb-4 w-48"
          >
            Story Crafter
          </button>
          <button
            onClick={() => setSelectedApp('recipe')}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 w-48"
          >
            Food Recipes
          </button>
        </div>
      ) : selectedApp === 'story' ? (
        <StoryCrafter />
      ) : (
        <FoodRecipes />
      )}

      {selectedApp && (
        <button
          onClick={() => setSelectedApp(null)}
          className="mt-8 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Back to Menu
        </button>
      )}
    </div>
  );
}
