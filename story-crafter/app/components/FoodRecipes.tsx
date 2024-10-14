'use client';

import { useState } from 'react';

export default function FoodRecipes() {
  const [food, setFood] = useState('');
  const [recipe, setRecipe] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGetRecipe = async () => {
    setError('');
    setRecipe('');
    setLoading(true);
    try {
      const res = await fetch('/api/recipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ food }),
      });

      const data = await res.json();

      if (res.ok) {
        setRecipe(data.recipe);
      } else {
        setError(data.error || 'An error occurred while fetching the recipe.');
      }
    } catch (error) {
      console.error('Error fetching recipe:', error);
      setError('An error occurred while fetching the recipe.');
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center w-full max-w-md">
      <h2 className="text-2xl font-bold mb-4">Food Recipes</h2>
      <input
        className="w-full p-2 text-black border border-gray-300 rounded mb-4"
        value={food}
        placeholder="Enter a food you want to cook..."
        onChange={(e) => setFood(e.target.value)}
      />
      {error && <p className="text-red-500 mb-4">{error}</p>}
      <button
        onClick={handleGetRecipe}
        className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        disabled={loading}
      >
        {loading ? 'Loading...' : 'Get Recipe'}
      </button>
      {recipe && (
        <div className="mt-4 whitespace-pre-wrap">
          <h3 className="text-xl font-semibold mb-2">Recipe:</h3>
          {recipe}
        </div>
      )}
    </div>
  );
}
