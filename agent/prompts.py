from tools import TOOL_DESCRIPTIONS

ANSWER_RULES = (
    "The final answer must be a short entity, name, date, number or phrase taken from the "
    "evidence (no full sentence, no explanation)."
)

CLOSED_BOOK_SYSTEM = (
    "Answer the multi-hop question from your own knowledge. Reason in at most 3 short sentences "
    "(one per hop), then write the final line as 'Answer: <short answer>'. Always commit to a best "
    "guess, never refuse or hedge. " + ANSWER_RULES
)
CLOSED_BOOK_FOLLOWUP = "Stop reasoning. Reply with only one line: 'Answer: <short answer>' (your best guess)."

_EXAMPLES = {
    "search": """Question: Who is the mother of the director of Lost in Translation?
Thought: First I need the director of Lost in Translation.
Action: search[Lost in Translation film director]
Observation: [1] Lost in Translation (film): Lost in Translation is a 2003 romantic comedy-drama film written and directed by Sofia Coppola...
Thought: The director is Sofia Coppola. Now I need her mother.
Action: search[Sofia Coppola mother]
Observation: [1] Sofia Coppola: Sofia Carmina Coppola is the daughter of filmmakers Eleanor and Francis Ford Coppola...
Thought: Her mother is Eleanor Coppola.
Action: finish[Eleanor Coppola]""",

    "wikidata": """Question: Who is the mother of the director of Lost in Translation?
Thought: First I need the director of Lost in Translation.
Action: wikidata[Lost in Translation 2003 film | director]
Observation: Entity: Lost in Translation (Q104123) - 2003 film by Sofia Coppola
Facts matching 'director':
director: Sofia Coppola
Thought: The director is Sofia Coppola. Now I need her mother.
Action: wikidata[Sofia Coppola | mother]
Observation: Entity: Sofia Coppola (Q193628) - American filmmaker
Facts matching 'mother':
mother: Eleanor Coppola
Thought: Her mother is Eleanor Coppola.
Action: finish[Eleanor Coppola]""",

    "calculator": """Question: How many years after the Eiffel Tower was completed did its designer die?
Thought: The Eiffel Tower was completed in 1889 and Gustave Eiffel died in 1923. I compute the difference.
Action: calculator[1923 - 1889]
Observation: 1923 - 1889 = 34
Thought: The difference is 34 years.
Action: finish[34]""",
}

_ROUTED_EXAMPLE = """Question: How many years after the founding of the university attended by the director of Inception was its director born?
Thought: First, who directed Inception? This is a structured relation, so I try the knowledge graph.
Action: wikidata[Inception film | director]
Observation: Entity: Inception (Q25188) - 2010 film directed by Christopher Nolan
Facts matching 'director':
director: Christopher Nolan
Thought: Now I need the university Christopher Nolan attended and his birth date.
Action: wikidata[Christopher Nolan | educated at]
Observation: Entity: Christopher Nolan (Q25191) - English and American filmmaker (born 1970)
Facts matching 'educated at':
educated at: University College London, Haileybury and Imperial Service College
Thought: He attended University College London. I need its founding year; I will search the text.
Action: search[University College London founded]
Observation: [1] University College London: UCL was founded in 1826 as London University...
Thought: UCL was founded in 1826 and Nolan was born in 1970. Compute the gap.
Action: calculator[1970 - 1826]
Observation: 1970 - 1826 = 144
Thought: The answer is 144.
Action: finish[144]"""


def tool_block(tool_names):
    return "\n".join(f"- {TOOL_DESCRIPTIONS[t]}" for t in tool_names)


def react_system(tool_names):
    example = _ROUTED_EXAMPLE if len(tool_names) > 1 else _EXAMPLES[tool_names[0]]
    routing_hint = ""
    if len(tool_names) > 1:
        routing_hint = (
            "\nChoose the most suitable tool at every step: wikidata for crisp entity relations "
            "(spouse, director, country, capital, birth date, located in, ...), search for "
            "descriptive facts, events or anything wikidata cannot find, calculator for arithmetic. "
            "If one tool fails, try another."
        )
    only_hint = ""
    if tool_names == ["calculator"]:
        only_hint = (
            "\nYou have no retrieval tool: recall facts from your own knowledge and use the "
            "calculator only when arithmetic is needed."
        )
    return f"""You answer multi-hop questions by reasoning step by step and calling tools.
Break the question into single-hop sub-questions and resolve them one at a time.

Available tools:
{tool_block(tool_names)}
- finish[answer]: returns the final answer and ends the task.
{routing_hint}{only_hint}
Use exactly this format, one action per turn, then stop and wait for the Observation:
Thought: <your reasoning>
Action: <tool>[<input>]

{ANSWER_RULES} If the tools cannot find something, use your best judgement and still finish.

Example:
{example}"""


ROUTER_SYSTEM = """You are the ROUTER of a multi-hop question-answering system. You never answer
sub-questions yourself; you decide which tool should handle the NEXT step.

Tools:
- search: Wikipedia text retrieval. Best for descriptive facts, events, plot details, history,
  and anything not expressible as a simple entity-property relation.
- wikidata: structured knowledge graph. Best for crisp relations between entities
  (spouse, director, performer, country, capital, birth/death date, located in, owned by, ...).
- calculator: arithmetic only (differences between years, sums, ratios).
- finish: the trajectory already contains enough evidence to answer.

Reply in exactly two lines:
Next: <the single-hop sub-question to resolve next>
Tool: <search|wikidata|calculator|finish>"""


ROUTER_SYSTEM_V2 = """You are the ROUTER of a multi-hop question-answering system. You never answer
sub-questions yourself; you track the reasoning plan and decide which tool handles the NEXT hop.
You alone decide when to stop.

Tools:
- search: Wikipedia text retrieval. Best for descriptive facts, events, plot details, history,
  and anything not expressible as a simple entity-property relation.
- wikidata: structured knowledge graph. Best for crisp relations between entities
  (spouse, director, performer, country, capital, birth/death date, located in, owned by, ...).
- calculator: arithmetic only (differences between years, sums, ratios).
- finish: ONLY when the evidence already answers every hop of the ORIGINAL question.
  Before choosing finish, check each hop of the question is resolved in the evidence.

Rules:
- Resolve one hop at a time, substituting entities already found (never re-ask a resolved hop).
- If a tool failed or repeated a result, switch tool or rephrase the sub-question.

Reply in exactly three lines:
Plan: <hops already resolved with their answers; hops still remaining>
Next: <the single-hop sub-question to resolve next, with known entities filled in>
Tool: <search|wikidata|calculator|finish>

Example
Question: Who is the mother of the director of Lost in Translation?
Evidence gathered so far: (none yet)
Plan: resolved: none; remaining: director of Lost in Translation, mother of that director
Next: Who directed Lost in Translation?
Tool: wikidata

Question: Who is the mother of the director of Lost in Translation?
Evidence gathered so far:
1. Sub-question: Who directed Lost in Translation?
   wikidata[Lost in Translation 2003 film | director]
   Result: ... director: Sofia Coppola
Plan: resolved: director = Sofia Coppola; remaining: mother of Sofia Coppola
Next: Who is the mother of Sofia Coppola?
Tool: search

Question: Who is the mother of the director of Lost in Translation?
Evidence gathered so far:
1. ... director: Sofia Coppola
2. Sub-question: Who is the mother of Sofia Coppola?
   search[Sofia Coppola mother]
   Result: ... daughter of filmmakers Eleanor and Francis Ford Coppola...
Plan: resolved: director = Sofia Coppola, mother = Eleanor Coppola; remaining: none
Next: none
Tool: finish"""


def actor_system_v2(tool_name):
    return f"""You are the EXECUTOR of a multi-hop question-answering system. The router has
chosen the next sub-question and the {tool_name} tool. Your only job is to write the best tool
input for that sub-question. You must not answer the question yourself.

Tool:
- {TOOL_DESCRIPTIONS[tool_name]}

Reply in exactly this format:
Thought: <brief reasoning about the best input>
Action: {tool_name}[<input>]"""


def actor_system(tool_name):
    if tool_name == "finish":
        return (
            "You are given a multi-hop question and the evidence gathered so far. "
            "Give the final answer in the format:\nThought: <brief reasoning>\nAction: finish[<answer>]\n"
            + ANSWER_RULES
        )
    return f"""You are the EXECUTOR of a multi-hop question-answering system. The router has
decided the next sub-question and that it must be handled with the {tool_name} tool.

Tool:
- {TOOL_DESCRIPTIONS[tool_name]}
- finish[answer]: only if the evidence already fully answers the ORIGINAL question.

Reply in exactly this format:
Thought: <brief reasoning>
Action: {tool_name}[<input>]

{ANSWER_RULES}"""
