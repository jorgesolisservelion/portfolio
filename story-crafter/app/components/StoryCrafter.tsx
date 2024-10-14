'use client';

import { useState } from 'react';

export default function StoryCrafter() {
  const [topic, setTopic] = useState('');
  const [storySoFar, setStorySoFar] = useState('');
  const [currentText, setCurrentText] = useState('');
  const [options, setOptions] = useState<string[]>([]);
  const [iteration, setIteration] = useState(0);
  const [completed, setCompleted] = useState(false);

  const handleStart = async () => {
    if (!topic) return;

    try {
      const res = await fetch('/api/story', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, iteration }),
      });

      const data = await res.json();

      if (res.ok) {
        const { story, options } = parseResponse(data.text);
        setStorySoFar(story);
        setCurrentText(story);
        setOptions(options);
        setIteration(1);
      } else {
        alert('An error occurred while generating the story.');
      }
    } catch (error) {
      console.error('Error starting story:', error);
      alert('An error occurred while generating the story.');
    }
  };

  const handleOptionSelect = async (index: number) => {
    const selectedOption = options[index];

    try {
      const res = await fetch('/api/story', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          storySoFar: storySoFar,
          choice: selectedOption,
          iteration,
        }),
      });

      const data = await res.json();

      if (res.ok) {
        const { story, options: newOptions } = parseResponse(data.text);
        setStorySoFar(prevStory => prevStory + '\n\n' + story);
        setCurrentText(story);
        setOptions(newOptions);
        setIteration(prev => prev + 1);

        if (iteration >= 5) {
          setCompleted(true);
        }
      } else {
        alert('An error occurred while generating the story.');
      }
    } catch (error) {
      console.error('Error continuing story:', error);
      alert('An error occurred while generating the story.');
    }
  };

  const parseResponse = (text: string) => {
    const [story, optionsText] = text.split('Options:');
    const options = optionsText
      .split('\n')
      .filter(line => line.trim())
      .map(line => line.replace(/^\d+\.\s*/, '').trim());
    return { story: story.replace('Story:', '').trim(), options };
  };

  const resetGame = () => {
    setTopic('');
    setStorySoFar('');
    setCurrentText('');
    setOptions([]);
    setIteration(0);
    setCompleted(false);
  };

  return (
    <div className="flex flex-col w-full max-w-md">
      <h2 className="text-2xl font-bold mb-4">Story Crafter</h2>
      {!completed ? (
        <>
          {iteration === 0 ? (
            <div>
              <input
                className="w-full p-2 text-black border border-gray-300 rounded mb-4"
                value={topic}
                placeholder="Enter a topic for your story..."
                onChange={(e) => setTopic(e.target.value)}
              />
              <button
                onClick={handleStart}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Start Story
              </button>
            </div>
          ) : (
            <div>
              <div className="mb-4">
                <h2 className="text-lg font-semibold">Story:</h2>
                <p className="whitespace-pre-wrap">{currentText}</p>
              </div>
              <div>
                <h2 className="text-lg font-semibold mb-2">Choose an option:</h2>
                {options.map((option, index) => (
                  <button
                    key={index}
                    onClick={() => handleOptionSelect(index)}
                    className="block w-full text-left px-4 py-2 mb-2 border border-gray-300 rounded hover:bg-gray-700 transition-colors duration-200"
                  >
                    {index + 1}. {option}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div>
          <h1 className="text-xl font-bold mb-4">Your Complete Story:</h1>
          <div className="whitespace-pre-wrap mb-6">{storySoFar}</div>
          <button
            onClick={resetGame}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Play Again
          </button>
        </div>
      )}
    </div>
  );
}
