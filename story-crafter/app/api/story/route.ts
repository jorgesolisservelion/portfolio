// app/api/story/route.ts

import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

// Configure the OpenAI API client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: NextRequest) {
  try {
    const { topic, storySoFar, choice, iteration } = await req.json();

    if (iteration === undefined) {
      return NextResponse.json({ error: 'Missing iteration parameter' }, { status: 400 });
    }

    let prompt = '';

    if (iteration === 0) {
      // Initial prompt
      prompt = `You are a creative and imaginative writer. Craft an engaging and entertaining story based on the following topic: "${topic}". The story should be one short paragraph, written in a fun, interesting, and easy-to-read style. **Use simple vocabulary suitable for a child around 10 years old. Avoid complex words and sentences.** Then, provide 4 numbered options (1 to 4) for the next part of the story that the user can choose from. Each option should be a single, concise sentence that intrigues the reader. Do not make the options too long or detailed. Format your response as:

Story:
[Your story]

Options:
1. [Option 1]
2. [Option 2]
3. [Option 3]
4. [Option 4]`;
    } else {
      // Subsequent prompts
      prompt = `You are a creative and imaginative writer. Continue the following story in a fun, interesting, and engaging way:

"${storySoFar}"

Based on the user's choice: "${choice}", write the next part of the story (one paragraph or less). **Use simple vocabulary suitable for a child around 10 years old. Avoid complex words and sentences.** Then, provide 4 new numbered options (1 to 4) for the user to choose from for the next part of the story. Each option should be a single, concise sentence that intrigues the reader. Do not make the options too long or detailed. Format your response as:

Story:
[Your story]

Options:
1. [Option 1]
2. [Option 2]
3. [Option 3]
4. [Option 4]`;
    }

    // Call the OpenAI API
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini-2024-07-18",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 500,
      temperature: 0.7,
    });

    const text = completion.choices[0].message.content.trim();

    return NextResponse.json({ text });
  } catch (error) {
    console.error('Error in API route:', error);
    return NextResponse.json({ error: 'An error occurred while processing your request.' }, { status: 500 });
  }
}
