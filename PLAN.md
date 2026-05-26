# תכנון משימה 1 - סוכן אנליסט נתונים לשירות לקוחות

## Context (רקע)

המטלה היא לבנות סוכן ReAct מבוסס LangGraph שיודע לענות על שאלות לגבי מאגר Bitext Customer Service (טבלת CSV עם 27K זוגות שאלה/תשובה של שירות לקוחות, 10 קטגוריות, 27 intents). הסוכן צריך לטפל ב-3 סוגי שאלות: structured (ספירות/רשימות), unstructured (סיכומים), out-of-scope (לדחות בנימוס).

הפרויקט הוא greenfield - לא קיים קוד עדיין. התיקייה מכילה רק את ה-PDF של המטלה.

**משימה 1 שווה 50 נקודות** ומתחלקת ל:
- Query router (15)
- Tools עם תיאורים ברורים ו-Pydantic schemas (15)
- Multi-step reasoning (10)
- CLI עם reasoning output (5)
- Max iterations fallback (5)

המטרה של התכנון הזה היא לחלק את משימה 1 לשלבים עצמאיים שאפשר לממש אחד-אחד, ולוודא שכל שלב מסתיים במשהו שאפשר לבדוק ולהריץ.

---

## החלטות מפתח

### בחירת מודלים - אסטרטגיית 2 מודלים

**מודל ה-Router**: `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`
- **למה**: הסיווג ל-structured/unstructured/out-of-scope הוא משימה פשוטה - מודל 8B מספיק בהחלט. ה-`-fast` הוא גרסה ב-fp8 quantization שמהירה משמעותית.
- **יתרון**: latency נמוך + עלות נמוכה לכל קריאה.

**מודל הסוכן (Agent + סיכומים)**: `meta-llama/Meta-Llama-3.1-70B-Instruct`
- **למה**: שאלות unstructured דורשות סיכום איכותי של מאות שורות; multi-step reasoning דורש בחירת כלים נכונה ושרשור שלהם. מודל 70B חזק משמעותית במשימות האלה.
- **יתרון**: איכות תשובות וסיכומים גבוהה משמעותית, פחות שגיאות בבחירת tools.

**ההצדקה לכותב ב-README**: המטלה רומזת ישירות לגישה זו ("If you use different models for different roles ... explain that"). אנחנו מקבלים את הטוב משני העולמות - מהירות ועלות נמוכה ב-router, ואיכות גבוהה ב-agent.

### בחירת ארכיטקטורת גרף

**גישה**: `create_react_agent` עטוף ב-Router node מותאם אישית.

- LangGraph מספקת פונקציה מוכנה `create_react_agent` שמטפלת בלולאת ReAct (Think → Act → Observe) באופן אוטומטי, כולל קריאות לכלים ו-max iterations.
- אנחנו לא בונים את הלולאה מאפס, אלא **עוטפים אותה** בגרף משלנו: Router נכנס ראשון, ומחליט אם להעביר ל-ReAct agent או ל-Decline node.

**למה לא לבנות הכל מאפס**:
- ReAct loop נכון הוא לא טריוויאלי (טיפול ב-tool errors, max iterations, message accumulation)
- עדיף להשקיע זמן בכלים הספציפיים שלנו ובניית ה-router, לא בלהמציא גלגל
- הציון מתבסס על האיכות של ה-router, הכלים והחשיבה - לא על שכתוב ReAct מאפס

### מבנה הגרף הסופי

```
       [START]
          ↓
      [Router]  ← מסווג את השאלה
          ↓
   ╱──────┼──────╲
   ↓      ↓       ↓
[Decline] [Structured] [Unstructured]
   ↓        ↓             ↓
   ↓     [ReAct Agent with tools]
   ↓        ↓             ↓
   ╲────────┼─────────────╱
            ↓
          [END]
```

הערה: structured ו-unstructured שניהם הולכים ל-ReAct agent (אותו loop, אותם tools), אבל ה-system prompt מותאם לסוג השאלה. ההפרדה בין השניים היא בעיקר לצורך הרציונל של ה-router והעדפות הסיכום.

---

## מבנה התיקיות

```
naomi submission/
├── data/
│   └── bitext_dataset.csv          # יורד מ-Hugging Face
├── src/
│   ├── __init__.py
│   ├── config.py                   # קונפיגורציה: model names, paths, max_iterations
│   ├── data_loader.py              # טעינת ה-CSV ל-DataFrame גלובלי
│   ├── tools.py                    # כל הכלים + Pydantic schemas
│   ├── router.py                   # Router node
│   ├── agent.py                    # בניית הגרף ב-LangGraph
│   └── cli.py                      # ממשק שורת פקודה
├── main.py                          # entry point
├── requirements.txt
├── .env.example                     # תבנית למפתח ה-API
├── .env                             # הקובץ האמיתי (gitignored)
├── .gitignore
└── README.md
```

---

## חלוקה לשלבים

כל שלב מסתיים במשהו רץ ובדיק לפני שעוברים הלאה.

### שלב 1: הכנת סביבה ותלויות

**מטרה**: סביבת עבודה מוכנה.

- ליצור `requirements.txt` עם:
  - `langgraph` (לגרף)
  - `langchain-openai` (חיבור ל-Nebius דרך OpenAI-compatible API)
  - `pandas` (לטעון את ה-CSV)
  - `pydantic` (לסכמות של כלים)
  - `python-dotenv` (לטעון `.env`)
  - `datasets` (להוריד מ-Hugging Face) או `huggingface_hub`
- ליצור virtual environment + `pip install -r requirements.txt`
- ליצור `.env.example` עם `NEBIUS_API_KEY=` ו-`.env` עם המפתח האמיתי
- ליצור `.gitignore` שמתעלם מ-`.env` ו-`data/`

**בדיקה**: `python -c "import langgraph, pandas, pydantic"` רץ בלי שגיאות.

---

### שלב 2: טעינת הדאטה והכרתו

**מטרה**: לוודא שהדאטה זמין ומבינים מה יש בו.

קובץ: `src/data_loader.py`

- פונקציה `load_dataset() -> pd.DataFrame` שמורידה (אם לא קיים) ומחזירה את ה-DataFrame
- **שימוש ב-`pd.read_csv()` ישירות מ-URL של Hugging Face** (במקום ספריית `datasets`).
  - הסיבה: ספריית `datasets` משתמשת ב-HF Hub עם rate limits לבקשות לא-מאומתות, מה שגורם לתקיעת ההורדה. הורדה ישירה דרך pandas לא סובלת מבעיה זו.
  - URL: `https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- שמירה מקומית ל-`data/bitext_dataset.csv` כדי לא להוריד כל הפעלה
- לוג בסיסי: כמה שורות, עמודות, רשימת קטגוריות ו-intents (לסניטי-צ'ק)

**בדיקה**: להריץ `python -c "from src.data_loader import load_dataset; df = load_dataset(); print(df.shape, df['category'].unique())"` ולראות שיוצא 5 עמודות, 10 קטגוריות.

---

### שלב 3: כלים (Tools) עם סכמות Pydantic

**מטרה**: סט כלים עצמאי שעובד על ה-DataFrame, עוד לפני שמחברים LLM.

קובץ: `src/tools.py`

הכלים המוצעים (מינימליים אבל מספיקים לכל שאלות הדוגמא):

1. **`list_categories()`** — מחזיר רשימת כל הקטגוריות הקיימות
2. **`list_intents(category: Optional[str])`** — רשימת כל ה-intents, אופציונלית מסונן לקטגוריה
3. **`count_rows(category: Optional[str], intent: Optional[str])`** — סופר שורות עם פילטרים אופציונליים
4. **`get_examples(n: int, category: Optional[str], intent: Optional[str])`** — מחזיר n דוגמאות (instruction + response) עם פילטרים
5. **`intent_distribution(category: str)`** — מחזיר dict של intent → count לקטגוריה
6. **`get_texts_for_summary(category: Optional[str], intent: Optional[str], n: int = 50)`** — שולף n שורות לסיכום (כלי ייעודי ל-unstructured queries)

**עקרון מנחה**: "A few well-designed tools beat many poorly described ones". 6 כלים זה מספיק - אפשר לשרשר אותם ל-multi-step reasoning.

לכל כלי:
- **Pydantic schema** של הקלט (משתמש ב-`pydantic.BaseModel`)
- **Docstring ברור** שמסביר מתי להשתמש בכלי, מה מקבלים, מה מקבלים בחזרה
- **Type hint** על ה-return value
- **Decorator** `@tool` של LangChain כדי להפוך אותו לכלי שה-LLM יכול לקרוא לו

**בדיקה**: כל כלי אפשר לקרוא לו ידנית מפייתון ולראות פלט הגיוני:
```python
from src.tools import count_rows
print(count_rows(intent="cancel_order"))  # מצופה ~1000
```

---

### שלב 4: חיבור ל-Nebius Token Factory

**מטרה**: לוודא ש-LLM עובד.

קובץ: `src/config.py`

- `ROUTER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast"`
- `AGENT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"`
- `MAX_ITERATIONS = 12`
- `NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1/"` (או `api.tokenfactory.nebius.com/v1/`)

יצירת מודלים ב-`src/agent.py` עם `ChatOpenAI`:
```python
ChatOpenAI(model=AGENT_MODEL, base_url=NEBIUS_BASE_URL, api_key=os.getenv("NEBIUS_API_KEY"))
```

**בדיקה**: סקריפט קצר שעושה `llm.invoke("Say hello")` ומוודא שיש תשובה.

---

### שלב 5: Router Node

**מטרה**: סיווג שאלות לפני העברה לסוכן.

קובץ: `src/router.py`

- פונקציה `route_query(state) -> str` שמשתמשת ב-`ROUTER_MODEL` עם prompt structured output
- ה-prompt מבקש מהמודל להחזיר אחד מ-3 ערכים: `structured` / `unstructured` / `out_of_scope`
- שימוש ב-`with_structured_output(Pydantic schema)` של LangChain לאכיפת התשובה
- הסכמה: `class RouteDecision(BaseModel): category: Literal["structured", "unstructured", "out_of_scope"]; reason: str`

**הנחיות ל-router ב-prompt**:
- structured = שאלה עם תשובה מספרית/רשימה/דוגמאות מהדאטה
- unstructured = שאלה פתוחה שדורשת סיכום של טקסטים מהדאטה
- out_of_scope = שאלה לא קשורה לדאטה של שירות לקוחות

**בדיקה ידנית**: להריץ על 6-8 שאלות דוגמא ולוודא שהסיווג נכון.

---

### שלב 6: בניית הגרף (Decline + ReAct Agent)

**מטרה**: גרף שלם שעובד end-to-end.

קובץ: `src/agent.py`

- **State**: `MessagesState` של LangGraph (רשימת מסרים)
- **Decline node**: פונקציה שמוסיפה הודעה: "I'm a customer service data analyst agent and can only answer questions about the Bitext dataset. Please ask me about categories, intents, examples, or summaries from the data."
- **Agent node**: שימוש ב-`create_react_agent(llm=AGENT_MODEL, tools=[...], state_schema=MessagesState)`
  - הגדרת `recursion_limit=MAX_ITERATIONS * 2` (כי כל איטרציה זה Think + Act)
- **Conditional edges מה-Router**:
  - `out_of_scope` → Decline → END
  - `structured` או `unstructured` → ReAct Agent → END
- **System prompt לסוכן** מותאם: מתאר את הדאטה, את הכלים, ומעודד שרשור כלים (multi-step reasoning)

**Max iterations fallback**:
- עטיפת הקריאה ל-graph ב-try/except על `GraphRecursionError`
- במקרה של חריגה: להחזיר הודעה "I couldn't reach a final answer within the iteration limit. Could you rephrase your question or break it into smaller parts?"

**בדיקה**: סקריפט קצר שמריץ שאילתה אחת ומדפיס את המסר הסופי.

---

### שלב 7: CLI אינטראקטיבי עם הדפסת reasoning

**מטרה**: ממשק שימוש בפועל.

קבצים: `src/cli.py`, `main.py`

- `main.py` הוא entry point קצר שקורא ל-`run_cli()` מ-`src/cli.py`
- לולאת `while True` עם `input("You: ")`
- שימוש ב-`graph.stream(..., stream_mode="updates")` כדי לקבל אירועים בזמן אמת
- בכל אירוע - להדפיס:
  - **Router decision** + ה-reason
  - **Tool call** עם שם הכלי + הארגומנטים
  - **Tool observation** (התוצאה, מקוצרת אם ארוכה)
  - **Final answer** של ה-LLM
- צביעת הפלט עם ANSI codes (אופציונלי, נראה יפה במסוף)
- פקודות מיוחדות: `/exit`, `/quit` ליציאה

**בדיקה**: להריץ `python main.py` ולעבור על 8 השאלות מהמטלה.

---

### שלב 8: בדיקות סופיות + תיעוד

**מטרה**: לוודא שהכל עובד מקצה לקצה.

- להריץ את כל 8 שאלות הדוגמא במטלה:
  1. "What categories exist in the dataset?" (structured, כלי אחד)
  2. "How many refund requests did we get?" (structured, **multi-step**: filter → count)
  3. "Show me 5 examples of the SHIPPING category." (structured)
  4. "Summarize how agents respond to complaint intents." (unstructured)
  5. "Show me examples of people wanting their money back." (structured, צריך mapping סמנטי ל-`track_refund`/`check_refund_policy`)
  6. "What is the distribution of intents in the ACCOUNT category?" (structured)
  7. "What's the best CRM software for handling complaints?" (out-of-scope)
  8. "Who is the president of France?" (out-of-scope)
- לוודא ש-multi-step reasoning קורה (שאלה 2 צריכה לראות שתי קריאות כלים)
- לוודא ש-out-of-scope נדחה בלי לענות מידע
- לבדוק את ה-max iterations fallback ידנית (אפשר זמנית להוריד את MAX_ITERATIONS ל-2)

עדכון `README.md` בסיסי שמכסה:
- Setup (clone, venv, pip install, .env)
- איך להריץ (`python main.py`)
- בחירת מודלים והנימוק
- רשימת הכלים
- (חלקי MCP ו-memory יתווספו במשימות 2-3)

---

## קבצים קריטיים

קבצים חדשים שייווצרו במהלך המימוש:

- `requirements.txt` — תלויות
- `.env.example`, `.gitignore` — תשתית
- `src/config.py` — קונפיגורציה מרוכזת
- `src/data_loader.py` — טעינת CSV
- `src/tools.py` — 6 כלים עם Pydantic schemas (הקובץ המרכזי לציון Tools)
- `src/router.py` — Router node (הקובץ המרכזי לציון Router)
- `src/agent.py` — בניית הגרף (multi-step reasoning + max iterations)
- `src/cli.py` — CLI עם הדפסת reasoning
- `main.py` — entry point
- `README.md` — תיעוד

---

## אימות סופי (Verification)

איך נדע שמשימה 1 הושלמה בהצלחה:

1. **`python main.py`** עולה ופותח prompt אינטראקטיבי
2. שאלה structured פשוטה ("What categories exist?") מחזירה רשימה נכונה של 10 קטגוריות
3. שאלה structured מורכבת ("How many refund requests?") מציגה **שתי קריאות tool** בפלט (multi-step reasoning הוכח)
4. שאלה unstructured ("Summarize FEEDBACK") מחזירה סיכום בשפה טבעית (לא ספירות)
5. שאלה out-of-scope ("Who is president of France?") מקבלת הודעת דחייה מנומסת **בלי תשובה אמיתית**
6. הגדרה זמנית של `MAX_ITERATIONS=2` גורמת להופעת ה-fallback message בשאלה מורכבת
7. בכל שאלה רואים בפלט: Router decision → Tool calls → Tool observations → Final answer

אם כל ה-7 עובדים → משימה 1 הושלמה (50/50 נקודות).
