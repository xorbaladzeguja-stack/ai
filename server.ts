import 'dotenv/config';
import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const GEMINI_API_KEY = (process.env.GEMINI_API_KEY || '').trim();
const HAS_GEMINI_KEY =
  GEMINI_API_KEY.length > 0 && GEMINI_API_KEY !== 'MY_GEMINI_API_KEY';

const ai = HAS_GEMINI_KEY
  ? new GoogleGenAI({
      apiKey: GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

// Web Push is optional and disabled until persistent VAPID keys and push storage are configured.
const WEB_PUSH_CONFIGURED = false;

// Deterministic arithmetic verifier for AI-generated math questions
function safeEvalArithmetic(expr: string): number | null {
  const cleaned = expr.replace(/×/g, '*').replace(/÷/g, '/').replace(/\s+/g, '');
  if (!/^[0-9+\-*/().]+$/.test(cleaned)) {
    return null;
  }
  try {
    // Tokenized recursive-descent parser (no eval / Function)
    let pos = 0;
    const parseExpr = (): number => {
      let left = parseTerm();
      while (pos < cleaned.length && (cleaned[pos] === '+' || cleaned[pos] === '-')) {
        const op = cleaned[pos++];
        const right = parseTerm();
        left = op === '+' ? left + right : left - right;
      }
      return left;
    };
    const parseTerm = (): number => {
      let left = parseFactor();
      while (pos < cleaned.length && (cleaned[pos] === '*' || cleaned[pos] === '/')) {
        const op = cleaned[pos++];
        const right = parseFactor();
        if (op === '/' && right === 0) throw new Error('Division by zero');
        left = op === '*' ? left * right : left / right;
      }
      return left;
    };
    const parseFactor = (): number => {
      if (cleaned[pos] === '(') {
        pos++;
        const val = parseExpr();
        if (cleaned[pos] !== ')') throw new Error('Mismatched parentheses');
        pos++;
        return val;
      }
      let sign = 1;
      if (cleaned[pos] === '-') {
        sign = -1;
        pos++;
      } else if (cleaned[pos] === '+') {
        pos++;
      }
      const start = pos;
      while (pos < cleaned.length && /[0-9.]/.test(cleaned[pos])) pos++;
      if (start === pos) throw new Error('Unexpected token');
      const num = Number(cleaned.slice(start, pos));
      if (!Number.isFinite(num)) throw new Error('Invalid number');
      return sign * num;
    };
    const result = parseExpr();
    if (pos !== cleaned.length || !Number.isFinite(result)) return null;
    return Math.round(result * 10000) / 10000;
  } catch {
    return null;
  }
}

function validateAndCorrectMathPractice(practice: {
  question?: string;
  expression?: string;
  options?: string[];
  correctAnswer?: string;
  explanation?: string;
}) {
  if (!practice || !practice.question || !practice.correctAnswer) return null;
  if (practice.expression) {
    const computed = safeEvalArithmetic(practice.expression);
    if (computed !== null) {
      const correctStr = String(computed);
      const opts = Array.isArray(practice.options) ? [...practice.options] : [];
      if (!opts.includes(correctStr)) {
        if (opts.length > 0) {
          opts[0] = correctStr;
        } else {
          opts.push(correctStr, String(computed + 1), String(computed + 2), String(Math.max(0, computed - 1)));
        }
      }
      return {
        ...practice,
        correctAnswer: correctStr,
        options: Array.from(new Set(opts)).slice(0, 4),
      };
    }
  }
  return practice;
}

const CRISIS_PATTERNS = [
  /თვითმკვლელ/i,
  /თავის\s*მოკვლ/i,
  /სიცოცხლე\s*აღარ\s*მინდა/i,
  /დავიზიან/i,
  /suicide/i,
  /kill\s*myself/i,
  /end\s*my\s*life/i,
  /self[- ]?harm/i,
];

function hasCrisisSignal(text: string): boolean {
  return CRISIS_PATTERNS.some((rx) => rx.test(text));
}

async function callGeminiWithRetry<T>(fn: () => Promise<T>, retries = 2): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (i < retries) {
        await new Promise((r) => setTimeout(r, 400 * (i + 1)));
      }
    }
  }
  throw lastError;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({
      ok: true,
      aiConfigured: HAS_GEMINI_KEY,
      webPushConfigured: WEB_PUSH_CONFIGURED,
      notificationChannels: ['in_app', 'browser_local'],
    });
  });

  app.post('/api/ai/teacher', async (req: Request, res: Response) => {
    if (!ai) {
      res.status(503).json({
        error: 'AI სერვისი ამჟამად მიუწვდომელია (GEMINI_API_KEY არ არის კონფიგურირებული). შეგიძლიათ გააგრძელოთ ოფლაინ გაკვეთილებით.',
        code: 'AI_UNAVAILABLE',
      });
      return;
    }

    const { message, mode = 'explain', subject = 'general', userContext = {}, history = [] } = req.body || {};
    if (typeof message !== 'string' || !message.trim()) {
      res.status(400).json({ error: 'შეიყვანეთ შეკითხვა.' });
      return;
    }

    const sanitizedMessage = message.trim().slice(0, 1500);
    const contextSummary = JSON.stringify({
      level: userContext.level || 'beginner',
      completedLessons: Number(userContext.completedLessons || 0),
      weakTopics: Array.isArray(userContext.weakTopics) ? userContext.weakTopics.slice(0, 6) : [],
      recentMistakes: Array.isArray(userContext.recentMistakes) ? userContext.recentMistakes.slice(0, 4) : [],
    });

    const modeInstructions: Record<string, string> = {
      direct:
        'მოსწავლემ აირჩია რეჟიმი „პირდაპირი ახსნა“. მიეცი მკაფიო, პირდაპირი, სრული და მარტივად გასაგები ქართული ახსნა ნაბიჯებითა და მაგალითით. შეგიძლია პირდაპირ აჩვენო სწორი პასუხიც, თუ კითხვა ამას მოითხოვს. დააყენე hintOnly: false.',
      explain:
        'მოსწავლემ აირჩია რეჟიმი „პირდაპირი ახსნა“. მიეცი მკაფიო, პირდაპირი, სრული და მარტივად გასაგები ქართული ახსნა ნაბიჯებითა და მაგალითით. დააყენე hintOnly: false.',
      step_by_step:
        'მოსწავლემ აირჩია რეჟიმი „ნაბიჯ-ნაბიჯ მინიშნებები“. კატეგორიულად აკრძალულია საბოლოო პასუხის პირდაპირ დაწერა! არ მისცე მზა პასუხი — დაუსვი 1 ან 2 მარტივი, მიმართულების მიმცემი შეკითხვა და მიეცი პატარა მინიშნება, რომ მოსწავლე თავად მივიდეს სწორ პასუხამდე. თუ მოსწავლემ უკვე სცადა პასუხი, შეაფასე და შემდეგი კითხვით მიიყვანე სწორ შედეგამდე. აუცილებლად დააყენე hintOnly: true.',
      hint:
        'მოსწავლემ აირჩია რეჟიმი „ნაბიჯ-ნაბიჯ მინიშნებები“. კატეგორიულად აკრძალულია საბოლოო პასუხის პირდაპირ დაწერა! დაუსვი 1–2 მარტივი შეკითხვა და მიეცი მხოლოდ მინიშნება. დააყენე hintOnly: true.',
      worked_example:
        'აჩვენე ეტაპობრივი ამოხსნილი მაგალითი (ნაბიჯი 1, ნაბიჯი 2...) და შემდეგ შესთავაზე მსგავსი სავარჯიშო კითხვა.',
      practice:
        'შექმენი 1 პრაქტიკული სავარჯიშო მოსწავლის დონის მიხედვით და მოკლე ახსნა.',
    };

    const systemInstruction = `შენ ხარ SmartLearn AI-ს პირადი მასწავლებელი.
წესები:
1. ყველა ახსნა დაწერე ბუნებრივი, მარტივი და მეგობრული ქართული ენით.
2. ინგლისურის სწავლებისას ახალი მასალა ყოველთვის ახსენი ქართულად.
3. ნუ იყენებ ყალბ ქებას; წაახალისე რეალისტურად და კონსტრუქციულად.
4. მკაცრად დაიცავი არჩეული რეჟიმის წესი (პირდაპირი ახსნა vs ნაბიჯ-ნაბიჯ მინიშნებები).
5. მოსწავლის კონტექსტი: ${contextSummary}. საგანი: ${String(subject)}.
6. რეჟიმის ინსტრუქცია: ${modeInstructions[String(mode)] || modeInstructions.direct}`;

    try {
      const response = await callGeminiWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: [
            ...(Array.isArray(history)
              ? history.slice(-6).map((h: { role?: string; text?: string }) => ({
                  role: h.role === 'model' ? 'model' : 'user',
                  parts: [{ text: String(h.text || '').slice(0, 600) }],
                }))
              : []),
            { role: 'user', parts: [{ text: sanitizedMessage }] },
          ],
          config: {
            systemInstruction,
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                reply: { type: Type.STRING, description: 'მთავარი ახსნა ან მინიშნება ქართულად.' },
                steps: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: 'ეტაპობრივი ნაბიჯები (თუ საჭიროა).',
                },
                hintOnly: { type: Type.BOOLEAN, description: 'true თუ პასუხი შეგნებულად დაფარულია მინიშნებისთვის.' },
                practiceQuestion: {
                  type: Type.OBJECT,
                  properties: {
                    question: { type: Type.STRING },
                    expression: { type: Type.STRING, description: 'თუ მათემატიკაა, მხოლოდ არითმეტიკული გამოსახულება მაგ. 7+5' },
                    options: { type: Type.ARRAY, items: { type: Type.STRING } },
                    correctAnswer: { type: Type.STRING },
                    explanation: { type: Type.STRING },
                  },
                },
              },
              required: ['reply', 'steps', 'hintOnly'],
            },
          },
        })
      );

      const parsed = JSON.parse((response.text || '{}').trim());
      if (parsed.practiceQuestion && subject === 'math') {
        const verified = validateAndCorrectMathPractice(parsed.practiceQuestion);
        if (verified) {
          parsed.practiceQuestion = verified;
        } else {
          delete parsed.practiceQuestion;
        }
      }
      res.json(parsed);
    } catch (err) {
      console.error('AI teacher error:', err);
      res.status(502).json({
        error: 'AI მასწავლებელთან დაკავშირება ვერ მოხერხდა. შეგიძლიათ გააგრძელოთ შენახული გაკვეთილებით.',
        code: 'AI_REQUEST_FAILED',
      });
    }
  });

  app.post('/api/ai/english', async (req: Request, res: Response) => {
    if (!ai) {
      res.status(503).json({
        error: 'AI სერვისი მიუწვდომელია. შეგიძლიათ გამოიყენოთ ოფლაინ ლექსიკონი და დიალოგები.',
        code: 'AI_UNAVAILABLE',
      });
      return;
    }

    const {
      userSentence = '',
      scenario = 'greeting',
      level = 'beginner',
      fromSpeech = false,
    } = req.body || {};
    if (typeof userSentence !== 'string' || !userSentence.trim()) {
      res.status(400).json({ error: 'გთხოვთ შეიყვანოთ ინგლისური წინადადება.' });
      return;
    }

    try {
      const response = await callGeminiWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `სცენარი: ${String(scenario)}. დონე: ${String(level)}. ხმოვანი ვარჯიში: ${
            fromSpeech ? 'დიახ' : 'არა'
          }. მოსწავლის ინგლისური ფრაზა: "${userSentence.trim().slice(0, 400)}"`,
          config: {
            systemInstruction:
              'შენ ხარ ინგლისური ენის ქართულენოვანი მასწავლებელი დამწყებისთვის. შეაფასე მოსწავლის ინგლისური წინადადება: 1) ქართულად აუხსენი გრამატიკა და სიტყვების წყობა (georgianFeedback); 2) მიეცი ქართული რჩევა გამოთქმისა და ფონეტიკის შესახებ (pronunciationTipKa — მაგ. ქართული ასოებით ჩაწერილი გამოთქმა და მახვილი); 3) დაწერე სწორი ინგლისური ვარიანტი (correctedEnglish); 4) გააგრძელე დიალოგი მარტივი ინგლისური ფრაზით + ქართული თარგმანით.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                isCorrect: { type: Type.BOOLEAN },
                correctedEnglish: { type: Type.STRING },
                georgianFeedback: { type: Type.STRING },
                pronunciationTipKa: { type: Type.STRING },
                nextEnglishPrompt: { type: Type.STRING },
                nextPromptGeorgian: { type: Type.STRING },
              },
              required: [
                'isCorrect',
                'correctedEnglish',
                'georgianFeedback',
                'pronunciationTipKa',
                'nextEnglishPrompt',
                'nextPromptGeorgian',
              ],
            },
          },
        })
      );

      res.json(JSON.parse((response.text || '{}').trim()));
    } catch (err) {
      console.error('AI english error:', err);
      res.status(502).json({
        error: 'ინგლისურის პრაქტიკის სერვისი დროებით მიუწვდომელია.',
        code: 'AI_REQUEST_FAILED',
      });
    }
  });

  app.post('/api/ai/weakness-quiz', async (req: Request, res: Response) => {
    if (!ai) {
      res.status(503).json({ error: 'AI მიუწვდომელია', code: 'AI_UNAVAILABLE' });
      return;
    }

    const { mistakesSummary = [], weakTopics = [], level = 'beginner' } = req.body || {};
    const safeMistakes = Array.isArray(mistakesSummary) ? mistakesSummary.slice(0, 8) : [];
    const safeTopics = Array.isArray(weakTopics) ? weakTopics.slice(0, 6) : [];

    try {
      const response = await callGeminiWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `მოსწავლის დონე: ${String(level)}.
სუსტი თემები: ${JSON.stringify(safeTopics)}.
ბოლო შეცდომები შეცდომების ბანკიდან: ${JSON.stringify(safeMistakes)}.
შექმენი ზუსტად 5 პერსონალიზებული სავარჯიშო ტესტი, რომელიც ამ შეცდომებისა და სუსტი მხარეების გაძლიერებაზეა ორიენტირებული.`,
          config: {
            systemInstruction:
              'შენ ხარ SmartLearn AI-ს პერსონალიზებული ტესტების შემდგენი. შექმენი ზუსტად 5 მკაფიო, სასწავლო ტესტური კითხვა მოსწავლის შეცდომებზე დაყრდნობით (თუ შეცდომები ცოტაა, დააბალანსე მარტივი მათემატიკითა და საბაზისო ინგლისურით). მათემატიკურ კითხვებში აუცილებლად შეავსე ველი expression (მხოლოდ არითმეტიკული გამოსახულება, მაგ. 8+7 ან 6*4). ყოველი კითხვისთვის მოამზადე ზუსტად 4 სავარაუდო პასუხი (options), სადაც ერთ-ერთი ზუსტად ემთხვევა correctAnswer-ს, და მოკლე ქართული ახსნა (explanation).',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  subject: { type: Type.STRING, description: 'math ან english' },
                  topic: { type: Type.STRING },
                  prompt: { type: Type.STRING },
                  expression: { type: Type.STRING },
                  options: { type: Type.ARRAY, items: { type: Type.STRING } },
                  correctAnswer: { type: Type.STRING },
                  explanation: { type: Type.STRING },
                },
                required: ['subject', 'topic', 'prompt', 'options', 'correctAnswer', 'explanation'],
              },
            },
          },
        })
      );

      const raw = JSON.parse((response.text || '[]').trim());
      const validated = Array.isArray(raw)
        ? raw
            .map((item, idx) => {
              if (!item || typeof item.prompt !== 'string' || typeof item.correctAnswer !== 'string') {
                return null;
              }
              if (item.subject === 'math' && typeof item.expression === 'string' && item.expression.trim()) {
                const checked = validateAndCorrectMathPractice({
                  question: item.prompt,
                  expression: item.expression,
                  options: item.options,
                  correctAnswer: item.correctAnswer,
                  explanation: item.explanation,
                });
                if (!checked || !Array.isArray(checked.options) || checked.options.length < 2) return null;
                return {
                  id: `wq_${Date.now()}_${idx}`,
                  subject: 'math' as const,
                  topic: String(item.topic || 'მათემატიკა'),
                  prompt: item.prompt,
                  options: checked.options,
                  correctAnswer: checked.correctAnswer,
                  explanation: String(item.explanation || ''),
                };
              }
              const opts = Array.isArray(item.options)
                ? item.options.map(String).filter(Boolean)
                : [];
              if (opts.length < 2 || !opts.includes(String(item.correctAnswer))) {
                return null;
              }
              return {
                id: `wq_${Date.now()}_${idx}`,
                subject: item.subject === 'math' ? ('math' as const) : ('english' as const),
                topic: String(item.topic || 'სუსტი თემა'),
                prompt: item.prompt,
                options: Array.from(new Set(opts)).slice(0, 4),
                correctAnswer: String(item.correctAnswer),
                explanation: String(item.explanation || ''),
              };
            })
            .filter(Boolean)
            .slice(0, 5)
        : [];

      res.json({ questions: validated });
    } catch (err) {
      console.error('AI weakness-quiz error:', err);
      res.status(502).json({
        error: 'სუსტი მხარეების ტესტის გენერაცია ვერ მოხერხდა.',
        code: 'AI_REQUEST_FAILED',
      });
    }
  });

  app.post('/api/ai/assignment', async (req: Request, res: Response) => {
    if (!ai) {
      res.status(503).json({ error: 'AI მიუწვდომელია', code: 'AI_UNAVAILABLE' });
      return;
    }

    const { level = 'beginner', weakTopics = [], subjects = ['math', 'english'] } = req.body || {};
    try {
      const response = await callGeminiWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `შექმენი 2 მოკლე სავარჯიშო კითხვა (დონე: ${String(level)}, საგნები: ${JSON.stringify(subjects)}, სუსტი თემები: ${JSON.stringify(weakTopics)}).`,
          config: {
            systemInstruction:
              'შექმენი 2 მკაფიო დავალება: 1 მათემატიკის (მხოლოდ მარტივი არითმეტიკა ველში expression, მაგ: 8+7) და 1 ინგლისურის (ქართული პირობით). ყველა ახსნა დაწერე ქართულად.',
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  subject: { type: Type.STRING },
                  prompt: { type: Type.STRING },
                  expression: { type: Type.STRING },
                  options: { type: Type.ARRAY, items: { type: Type.STRING } },
                  correctAnswer: { type: Type.STRING },
                  explanation: { type: Type.STRING },
                },
                required: ['subject', 'prompt', 'options', 'correctAnswer', 'explanation'],
              },
            },
          },
        })
      );

      const raw = JSON.parse((response.text || '[]').trim());
      const validated = Array.isArray(raw)
        ? raw
            .map((item) => {
              if (item.subject === 'math' && item.expression) {
                const checked = validateAndCorrectMathPractice({
                  question: item.prompt,
                  expression: item.expression,
                  options: item.options,
                  correctAnswer: item.correctAnswer,
                  explanation: item.explanation,
                });
                if (!checked) return null;
                return { ...item, options: checked.options, correctAnswer: checked.correctAnswer };
              }
              if (!Array.isArray(item.options) || !item.options.includes(item.correctAnswer)) {
                return null;
              }
              return item;
            })
            .filter(Boolean)
        : [];
      res.json({ questions: validated });
    } catch {
      res.status(502).json({ error: 'AI დავალების გენერაცია ვერ მოხერხდა', code: 'AI_REQUEST_FAILED' });
    }
  });

  app.post('/api/ai/wellness', async (req: Request, res: Response) => {
    const { message = '', mood = 'calm', history = [] } = req.body || {};
    if (typeof message !== 'string' || !message.trim()) {
      res.status(400).json({ error: 'შეიყვანეთ შეტყობინება.' });
      return;
    }

    const sanitized = message.trim().slice(0, 1200);
    if (hasCrisisSignal(sanitized)) {
      res.json({
        reply:
          'ძალიან ვწუხვარ, რომ ასეთ მძიმე მომენტს გადიხარ. თუ ახლა საფრთხეში ხარ ან საკუთარი თავის დაზიანებაზე ფიქრობ, გთხოვ დაუყოვნებლივ დაუკავშირდი გადაუდებელ დახმარებას (112 საქართველოში) ან სანდო ახლობელს/ოჯახის წევრს. შენ მარტო არ ხარ და პროფესიონალური დახმარება ხელმისაწვდომია.',
        safetyAlert: true,
        suggestedExercise: 'breathing',
      });
      return;
    }

    if (!ai) {
      res.status(503).json({
        error: 'AI მხარდამჭერი ამჟამად მიუწვდომელია. შეგიძლიათ გამოიყენოთ სუნთქვის სავარჯიშო და პირადი დღიური ოფლაინ რეჟიმში.',
        code: 'AI_UNAVAILABLE',
      });
      return;
    }

    try {
      const response = await callGeminiWithRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: [
            ...(Array.isArray(history)
              ? history.slice(-6).map((h: { role?: string; text?: string }) => ({
                  role: h.role === 'model' ? 'model' : 'user',
                  parts: [{ text: String(h.text || '').slice(0, 500) }],
                }))
              : []),
            { role: 'user', parts: [{ text: `მიმდინარე განწყობა: ${String(mood)}. შეტყობინება: ${sanitized}` }] },
          ],
          config: {
            systemInstruction: `შენ ხარ SmartLearn AI-ს თბილი, პატივისცემით სავსე ფსიქოლოგიური კეთილდღეობის დამხმარე.
უსაფრთხოების მკაცრი წესები:
1. არასოდეს დასვა სამედიცინო ან ფსიქიატრიული დიაგნოზი.
2. არასოდეს თქვა, რომ ცვლი ფსიქოლოგს ან ექიმს.
3. არ გასცე სახიფათო სამედიცინო რჩევა.
4. არ წაახალისო AI-ზე ემოციური დამოკიდებულება — წაახალისე რეალური ჩვევები, დასვენება და ადამიანებთან ურთიერთობა.
5. ისაუბრე თბილი, მშვიდი და არაგანმსჯელი ქართულით (მაქს. 4-5 წინადადება).`,
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                reply: { type: Type.STRING },
                suggestedExercise: {
                  type: Type.STRING,
                  description: 'ერთ-ერთი: breathing | journal | break | none',
                },
              },
              required: ['reply', 'suggestedExercise'],
            },
          },
        })
      );

      const parsed = JSON.parse((response.text || '{}').trim());
      res.json({ ...parsed, safetyAlert: false });
    } catch (err) {
      console.error('AI wellness error:', err);
      res.status(502).json({
        error: 'კავშირის შეცდომა. გამოიყენეთ სუნთქვის ტაიმერი ან პირადი დღიური.',
        code: 'AI_REQUEST_FAILED',
      });
    }
  });

  // Modular Web Push endpoints — non-blocking local-first response until Web Push is enabled later.
  app.post('/api/push/subscribe', (_req: Request, res: Response) => {
    res.json({
      ok: false,
      webPushConfigured: WEB_PUSH_CONFIGURED,
      mode: 'local_only',
    });
  });

  app.post('/api/push/unsubscribe', (_req: Request, res: Response) => {
    res.json({
      ok: true,
      webPushConfigured: WEB_PUSH_CONFIGURED,
    });
  });

  app.post('/api/push/test', (_req: Request, res: Response) => {
    res.json({
      ok: false,
      webPushConfigured: WEB_PUSH_CONFIGURED,
      mode: 'local_only',
    });
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`SmartLearn AI server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
