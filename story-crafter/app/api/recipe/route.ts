import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: NextRequest) {
  try {
    const { food } = await req.json();

    // First, check if the input is a food item
    const validationCompletion = await openai.chat.completions.create({
      model: "gpt-4o-mini-2024-07-18",
      messages: [
        { role: "system", content: "You are a helpful assistant that determines if an input is a food item or dish." },
        { role: "user", content: `Is "${food}" a food item or dish? Answer with only 'Yes' or 'No'.` }
      ],
      max_tokens: 5,
      temperature: 0,
    });

    const isFood = validationCompletion.choices[0].message.content.trim().toLowerCase() === 'yes';

    if (!isFood) {
      return NextResponse.json({ error: 'Please enter a valid food or dish name.' }, { status: 400 });
    }

    // If it is a food item, proceed with generating the recipe
    const recipeCompletion = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: "You are a helpful assistant that provides recipes." },
        { role: "user", content: `Provide a recipe for ${food}. Include a list of ingredients and step-by-step instructions.` }
      ],
      max_tokens: 500,
      temperature: 0.7,
    });

    const recipe = recipeCompletion.choices[0]?.message?.content?.trim() ?? 'No recipe generated.';

    return NextResponse.json({ recipe });
  } catch (error) {
    console.error('Error in API route:', error);
    return NextResponse.json({ error: 'An error occurred while processing your request.' }, { status: 500 });
  }
}
