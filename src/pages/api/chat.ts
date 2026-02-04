import type { APIRoute } from 'astro';
import OpenAI from "openai";
import { SecretsManagerClient, GetSecretValueCommand } from "@aws-sdk/client-secrets-manager";

export const prerender = false;

// Bishal's context - inspiring, connecting ideas to real work
const BISHAL_CONTEXT = `You are channeling Bishal Upadhyaya on his portfolio. When someone shares an idea, your job is to:
1. Get excited about their idea
2. Connect it to skills Bishal has PROVEN through real work
3. Point them toward relevant projects (Vibeset or research)
4. Fill them with confidence that Bishal can make it happen

BISHAL'S PROVEN WORK:
- Vibeset: AI music discovery that matches tracks to astrological vibes and moods. Built the full stack - LLM integration, audio processing, recommendation engine, beautiful UI.
- Research: Academic work in AI systems and intelligent applications.

BISHAL'S SKILLS (only mention what's relevant):
- LLM integration: GPT-4, Mistral, Gemini - building agents that actually understand context
- Music/Audio AI: Fingerprinting, mood matching, discovery systems  
- Recommendation engines: Personalized, context-aware suggestions
- Full-stack shipping: Python, TypeScript, AWS, serverless - prototype to production

TONE: Warm, confident, direct. Like a friend who genuinely wants to build something cool with them.
- Never sound like a service chatbot
- Keep it to 2-3 punchy sentences
- End with a specific connection to Vibeset or research
- Make them feel like their idea is totally buildable`;

const getSecret = async (secretName: string) => {
    // In Astro, use import.meta.env for access to .env variables
    const region = import.meta.env.AWS_DEFAULT_REGION || "us-east-2";
    const accessKeyId = import.meta.env.AWS_ID;
    const secretAccessKey = import.meta.env.AWS_SEC;

    // Fallback or explicit check
    if (!accessKeyId || !secretAccessKey) {
        console.warn("AWS Credentials missing in import.meta.env");
    }

    const client = new SecretsManagerClient({
        region,
        credentials: {
            accessKeyId: accessKeyId || "",
            secretAccessKey: secretAccessKey || "",
        },
    });

    try {
        const response = await client.send(
            new GetSecretValueCommand({
                SecretId: secretName,
            })
        );

        if (response.SecretString) {
            return JSON.parse(response.SecretString);
        }
    } catch (error) {
        console.error(`Failed to fetch secret ${secretName}:`, error);
    }
    return null;
};

export const POST: APIRoute = async ({ request }) => {
    try {
        const body = await request.json();
        const { message } = body;

        if (!message || typeof message !== "string") {
            return new Response(
                JSON.stringify({ error: "Message is required" }),
                { status: 400, headers: { "Content-Type": "application/json" } }
            );
        }

        // Try getting API key from environment first (fastest, supports local dev)
        let apiKey = import.meta.env.MISTRAL_API_KEY;

        // Fallback to AWS Secrets Manager if not in env
        if (!apiKey) {
            console.log("MISTRAL_API_KEY not in env, fetching from AWS Secrets Manager...");
            const secret = await getSecret("vibeset/azure_ai_foundry");
            apiKey = secret?.api_key_o3;
        }

        if (!apiKey) {
            console.error("Failed to retrieve API key from env or secrets manager");
            return new Response(
                JSON.stringify({
                    error: "Chat service not configured",
                    reply: "I'm not fully set up yet. Please contact Bishal directly at the contact page!"
                }),
                { status: 503, headers: { "Content-Type": "application/json" } }
            );
        }

        // Initialize OpenAI client with Azure Mistral endpoint
        const client = new OpenAI({
            baseURL: "https://kevin-m86fxp36-eastus2.services.ai.azure.com/openai/v1/",
            apiKey: apiKey,
        });

        // Call Mistral
        const response = await client.chat.completions.create({
            model: "mistral-small-2503",
            messages: [
                { role: "system", content: BISHAL_CONTEXT },
                { role: "user", content: message },
            ],
            temperature: 0.7,
            max_tokens: 200,
        });

        const reply = response.choices[0]?.message?.content || "I couldn't generate a response. Please try again.";

        return new Response(
            JSON.stringify({ reply }),
            { status: 200, headers: { "Content-Type": "application/json" } }
        );

    } catch (error) {
        console.error("Chat error:", error);
        return new Response(
            JSON.stringify({
                error: "Failed to process request",
                reply: "Something went wrong. Please try again or contact Bishal directly!"
            }),
            { status: 500, headers: { "Content-Type": "application/json" } }
        );
    }
};
