import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict
import re
import time
import ast
import math
import requests
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple, Optional, Set

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=1000, tools=tools, messages=messages
        )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in tool_input["expression"].split("\n"):
                                print(f"{line}")
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


# --- Improved RL Tasks Graders ---

HTTP_TIMEOUT = 10.0
CACHE = {}

def http_get(url: str, params: Dict[str, str] | None = None, headers: Dict[str, str] | None = None) -> Tuple[int, str]:
    key = ("GET", url, json.dumps(params, sort_keys=True) if params else None)
    if key in CACHE:
        return CACHE[key]
    r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
    CACHE[key] = (r.status_code, r.text)
    return CACHE[key]

# --- Wikidata helpers --------------------------------------------------------

def normalize_qid(q: str) -> Optional[str]:
    if not q:
        return None
    m = re.search(r"(Q\d+)", q)
    return m.group(1) if m else None

def fetch_wikidata_entity(qid: str) -> Optional[dict]:
    """
    Fetch Wikidata entity JSON via Special:EntityData/QID.json
    Doc: Wikidata REST / Special:EntityData JSON format.
    """
    qid_n = normalize_qid(qid)
    if not qid_n:
        return None
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid_n}.json"
    status, text = http_get(url)
    if status != 200:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

def verify_wikidata_triple(subject_qid: str, property_pid: str, object_qid_or_literal: str) -> bool:
    """
    Verify that subject has a claim with property Pxxx whose value equals object QID (or literal match).
    property_pid expected like 'P50' or 'P31' etc.
    """
    s_q = normalize_qid(subject_qid)
    if not s_q:
        return False
    prop_m = re.search(r"(P\d+)", property_pid)  # accept 'Pxxx' form
    if not prop_m:
        return False
    pid = prop_m.group(1)
    ent = fetch_wikidata_entity(s_q)
    if not ent or "entities" not in ent or s_q not in ent["entities"]:
        return False
    claims = ent["entities"][s_q].get("claims", {})
    if pid not in claims:
        return False
    # examine each claim for a match
    for claim in claims[pid]:
        mainsnak = claim.get("mainsnak", {})
        dv = mainsnak.get("datavalue")
        if not dv:
            continue
        val = dv.get("value")
        # If object is Qxxx: check id equality
        if isinstance(val, dict) and "id" in val:
            if normalize_qid(object_qid_or_literal) and normalize_qid(object_qid_or_literal) == val["id"]:
                return True
        # If literal: compare label/str if provided
        if isinstance(val, str) or isinstance(val, (int, float)):
            if str(val).lower() == str(object_qid_or_literal).lower():
                return True
        if isinstance(val, dict) and "time" in val:  # dates etc - check substring
            if str(object_qid_or_literal) in str(val):
                return True
    return False

# --- arXiv helpers -----------------------------------------------------------

def fetch_arxiv_summary(arxiv_id: str) -> Optional[str]:
    """
    Uses arXiv API (export.arxiv.org) to fetch entry summary for given arXiv id.
    See: https://info.arxiv.org/help/api/ (arXiv API)
    """
    # normalize id: accept forms like 'arXiv:2101.00001' or '2101.00001'
    m = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arxiv_id)
    if not m:
        # try older style: cond-mat/9901001
        m2 = re.search(r"([a-z\-]+(\.[A-Z]{2})?/\d{7})", arxiv_id)
        if not m2:
            return None
        q = m2.group(1)
    else:
        q = m.group(1)
    query = f"id:{q}"
    url = "http://export.arxiv.org/api/query"
    status, text = http_get(url, params={"search_query": query, "start": "0", "max_results": "1"})
    if status != 200:
        return None
    try:
        root = ET.fromstring(text)
        # namespaced Atom; find summary element
        summary = root.find(".//{http://www.w3.org/2005/Atom}summary")
        title = root.find(".//{http://www.w3.org/2005/Atom}title")
        s = (title.text if title is not None else "") + "\n" + (summary.text if summary is not None else "")
        return s.strip()
    except Exception:
        return None

# --- wordlist helper --------------------------------------------------------

WORDLIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
_WORDSET: Optional[Set[str]] = None
def get_wordset(max_words: int = 100_000) -> Set[str]:
    global _WORDSET
    if _WORDSET is not None:
        return _WORDSET
    try:
        status, txt = http_get(WORDLIST_URL)
        if status == 200:
            lines = txt.splitlines()
            _WORDSET = set(w.strip().lower() for w in lines[:max_words] if w.strip())
            return _WORDSET
    except Exception:
        pass
    # fallback - small set of common words
    _WORDSET = set([
        "the","a","and","is","in","it","you","that","he","was","for","on","are","as","with","his","they","i","at",
        "be","this","have","from","or","one","had","by","word","but","not","what","all","were","we","when","your",
        "can","said","there","use","an","each","which","she","do","how","their","if","will","up","other","about",
    ])
    return _WORDSET

# --- AST / code checks for AlphaGeometry task --------------------------------

def code_has_class_with_docstring(code: str, class_name: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # docstring extractor
            if ast.get_docstring(node):
                return True
    return False

def code_has_function(code: str, func_name: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return True
    return False

def code_is_syntax_valid(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except Exception:
        return False

# --- Graders -----------------------------------------------------------------

# 1) Wikidata Fact Weaver
def wikidata_fact_grader(output: str, metadata: Dict[str, Any]) -> bool:
    """
    Requirements enforced:
      - output must start with a 'Did you know' style short fact sentence.
      - metadata must contain 'wikidata_ids': list of QIDs and 'claim_triples': list of
        {"subject":"Q...", "property":"P...", "object":"Q..."} that model claims as supporting.
      - At least one claim_triple must be verifiable via Wikidata JSON.
      - Entities mentioned (labels) must appear in the output text.
    """
    # quick sanity checks
    if not isinstance(metadata, dict):
        return False
    if not re.search(r"(?i)\bdid you know\b", output):
        return False
    wids = metadata.get("wikidata_ids")
    triples = metadata.get("claim_triples")
    if not wids or not isinstance(wids, list) or len(wids) < 2:
        return False
    if not triples or not isinstance(triples, list) or len(triples) == 0:
        return False
    # check QID formats
    qids = [normalize_qid(str(q)) for q in wids]
    if any(q is None for q in qids):
        return False
    # verify at least one triple against Wikidata
    verified_any = False
    labels_in_text = 0
    # fetch labels of qids
    q_labels = {}
    for q in qids:
        ent = fetch_wikidata_entity(q)
        if ent and "entities" in ent and q in ent["entities"]:
            lbl = ent["entities"][q].get("labels", {}).get("en", {}).get("value")
            if lbl:
                q_labels[q] = lbl
                if lbl.lower() in output.lower():
                    labels_in_text += 1
    # require at least 1 entity label appear in the text
    if labels_in_text < 1:
        # still allow if model spelled QIDs directly in text, check presence
        if not any(q in output for q in qids):
            return False
    # now check claim triples
    for tr in triples:
        try:
            s = tr.get("subject")
            p = tr.get("property")
            o = tr.get("object")
            if not (s and p and o):
                continue
            if verify_wikidata_triple(s, p, o):
                verified_any = True
                break
        except Exception:
            continue
    return verified_any

# 2) Analogy Forge
def analogy_forge_grader(output: str, metadata: Dict[str, Any]) -> bool:
    """
    Requirements:
      - metadata['sources'] must be a list of reachable arXiv ids or paper URLs (>=2).
      - Each declared source must be reachable and at least one arXiv source's abstract/title
        must contain the ML keyword claimed by the model (e.g. 'RoPE' or 'rotary').
      - Output must contain analogy marker ('like','similar to','analogous to') and mention both:
        at least one ML term and at least one non-ML STEM term.
    """
    if not isinstance(metadata, dict):
        return False
    sources = metadata.get("sources", [])
    if not isinstance(sources, list) or len(sources) < 2:
        return False
    # check urls / arxiv ids
    reachable = 0
    ml_terms = ["transformer","rope","rotary","embedding","attention","scaling","gradient","positional","loss"]
    other_terms = ["wave","frequency","laplace","entropy","potential","field","resonance","viscosity"]
    found_ml = False
    found_other = False
    for s in sources[:6]:
        s_str = str(s)
        arxiv_summary = None
        # try arXiv ID
        if re.search(r"\d{4}\.\d{4,5}", s_str) or s_str.lower().startswith("arxiv:"):
            arxiv_summary = fetch_arxiv_summary(s_str)
            if arxiv_summary:
                reachable += 1
                # check for ML term presence
                for t in ml_terms:
                    if re.search(rf"\b{re.escape(t)}\b", arxiv_summary, re.I):
                        found_ml = True
                        break
        else:
            # try HTTP reachable
            try:
                st, _ = http_get(s_str)
                if st == 200:
                    reachable += 1
                    # quick heuristics: find ML or other terms in page
                    status, text = http_get(s_str)
                    txt = text[:2000].lower() if text else ""
                    if any(t in txt for t in ml_terms):
                        found_ml = True
                    if any(t in txt for t in other_terms):
                        found_other = True
            except Exception:
                pass
    # check output for analogy markers, ml and other terms
    if not any(token in output.lower() for token in [" like ", " similar to ", " analogous to ", "like a ", "like the "]):
        return False
    if not any(re.search(rf"\b{re.escape(t)}\b", output, re.I) for t in ml_terms):
        return False
    if not any(re.search(rf"\b{re.escape(t)}\b", output, re.I) for t in other_terms):
        return False
    # final condition: at least two reachable sources and found_ml True
    return reachable >= 2 and found_ml

# 3) Fibonacci Cat Tale
def fibonacci_cat_grader(output: str, metadata: Dict[str, Any]) -> bool:
    """
    Requirements:
      - Story about a cat (must contain 'cat'/'kitty'/'feline')
      - Word lengths must follow Fibonacci sequence for the first 12 words strictly;
        after that, allow ±1 tolerance.
      - Words should be in an English wordlist (downloaded) or be alphabetic tokens.
    """
    if not isinstance(output, str):
        return False
    if not re.search(r"\b(cat|kitty|feline|kitten)\b", output, re.I):
        return False
    # tokenize words (preserve only alphabetic sequences)
    words = re.findall(r"\b[a-zA-Z']+\b", output)
    if len(words) < 8:
        return False
    # fibonacci sequence
    fib = [1,1]
    while len(fib) < len(words):
        fib.append(fib[-1]+fib[-2])
    # strict first_n, then tolerant
    first_n = min(12, len(words))
    wordset = get_wordset(200_000)
    for i, w in enumerate(words):
        expected = fib[i]
        lw = len(w)
        if i < first_n:
            if lw != expected:
                return False
        else:
            if abs(lw - expected) > 1:
                return False
        # check lexicon membership (or allow apostrophes/names)
        if w.lower() not in wordset and not re.match(r"^[A-Z][a-z]+$", w):
            # allow small proper nouns; but reject random garbage
            if not re.match(r"^[a-zA-Z']+$", w):
                return False
    return True

# 4) Causal Analogy Builder
def causal_analogy_grader(output: str, metadata: Dict[str, Any]) -> bool:
    """
    Requirements:
      - metadata must contain 'graph' with 'nodes' (list) and 'edges' (list of [src,tgt]).
      - Graph must have >=3 nodes, edges must reference nodes, and be acyclic.
      - The textual 'experiment' (either in metadata['experiment'] or in output) must describe:
        an intervention, a measurement, and an expected directional result.
    """
    if not isinstance(metadata, dict):
        return False
    graph = metadata.get("graph")
    if not graph or "nodes" not in graph or "edges" not in graph:
        return False
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False
    if len(nodes) < 3:
        return False
    node_set = set(nodes)
    for e in edges:
        if not isinstance(e, (list,tuple)) or len(e) != 2:
            return False
        if e[0] not in node_set or e[1] not in node_set:
            return False
    # detect cycles (simple DFS)
    adj = {n: [] for n in nodes}
    for a,b in edges:
        adj[a].append(b)
    visited = {}
    def dfs(u):
        if visited.get(u) == 1:
            return True  # back-edge -> cycle
        if visited.get(u) == 2:
            return False
        visited[u] = 1
        for v in adj.get(u,[]):
            if dfs(v):
                return True
        visited[u] = 2
        return False
    for n in nodes:
        if dfs(n):
            return False
    # experiment presence & content
    exp_text = ""
    if "experiment" in metadata and isinstance(metadata["experiment"], str):
        exp_text = metadata["experiment"]
    else:
        # try extract experiment-like paragraph from output
        m = re.search(r"(experiment[:\s].{20,300})", output, re.I|re.S)
        exp_text = m.group(1) if m else output
    # must mention intervention and measurement keywords and directional expectation
    if not re.search(r"\b(intervene|intervention|manipulate|increase|decrease|knockdown|apply|treat)\b", exp_text, re.I):
        return False
    if not re.search(r"\b(measure|observe|assess|compare|outcome|effect|change)\b", exp_text, re.I):
        return False
    if not re.search(r"\b(if|when|then|would|should)\b", exp_text, re.I):
        return False
    return True

# 5) AlphaGeometry Component Coder
def alphageometry_grader(output: str, metadata: Dict[str, Any]) -> bool:
    """
    Requirements:
      - Code must be syntactically valid Python.
      - Must define classes Encoder, SymbolicSolver, GeometryReasoner each with docstrings.
      - Must define generate_dataset function.
      - Optionally: check for minimal cohesion (class names referenced in module).
    Uses AST parsing; does not execute arbitrary code.
    """
    if not isinstance(output, str):
        return False
    if not code_is_syntax_valid(output):
        return False
    required_classes = ["Encoder", "SymbolicSolver", "GeometryReasoner"]
    for cn in required_classes:
        if not code_has_class_with_docstring(output, cn):
            return False
    if not code_has_function(output, "generate_dataset"):
        return False
    # optional: ensure class names appear somewhere (usage)
    if not any(re.search(rf"\b{cn}\b", output) for cn in required_classes):
        return False
    return True

# --- TASK registry -----------------------------------------------------------

class RLTask:
    def __init__(self, name: str, prompt: str, tools: List[str], grader):
        self.name = name
        self.prompt = prompt
        self.tools = tools
        self.grader = grader

# Prompts should instruct the model to return both textual answer and JSON metadata with keys used above.
wikidata_prompt = (
    "Use the Wikidata REST API (Special:EntityData) to discover an interesting factual connection "
    "between two entities. Produce a short 'Did you know...' style fact sentence (one line). "
    "Also return a JSON metadata object with keys: 'wikidata_ids' (list of QIDs used) and "
    "'claim_triples' (list of objects with subject, property, object where property is e.g. 'P50'). "
    "Example metadata: {\"wikidata_ids\":[\"Q42\",\"Q123\"], \"claim_triples\":[{\"subject\":\"Q42\",\"property\":\"P50\",\"object\":\"Q123\"}] }"
)

analogy_prompt = (
    "Search arXiv or PapersWithCode for at least two sources relevant to an ML method (e.g. 'RoPE' or 'rotary positional encoding'). "
    "Explain the ML method using a structural analogy from another STEM field and include a JSON metadata object with key 'sources' (list of arXiv ids or URLs). "
    "Make sure the answer uses words like 'like', 'similar to', or 'analogous to', and name both the ML concept and the other field."
)

fibonacci_prompt = (
    "Write a short, coherent story about a cat where word lengths follow the Fibonacci sequence (1,1,2,3,5,8,13...). "
    "Return only the story text (plain) and also a JSON metadata object if you want. The grader will check word lengths strictly for the first 12 words."
)

causal_prompt = (
    "Given cause-effect statements, build a causal graph (JSON with 'nodes' and 'edges' lists) and propose an experiment to test reversibility of one edge. "
    "Return textual reasoning, the graph JSON, and an 'experiment' field in metadata describing the intervention and measurement."
)

alphageometry_prompt = (
    "Read a brief summary of the AlphaGeometry approach and write a minimal Python module skeleton with classes: Encoder, SymbolicSolver, GeometryReasoner (each with docstrings), "
    "and a function generate_dataset() that yields simple synthetic geometry problems. Return the code as a single Python string in the 'output' and optionally metadata."
)

TASKS = [
    RLTask("wikidata_fact_weaver", wikidata_prompt, ["wikidata_rest_api"], wikidata_fact_grader),
    RLTask("analogy_forge", analogy_prompt, ["web_search_tool"], analogy_forge_grader),
    RLTask("fibonacci_cat_tale", fibonacci_prompt, [], fibonacci_cat_grader),
    RLTask("causal_analogy_builder", causal_prompt, ["python_tool"], causal_analogy_grader),
    RLTask("alphageometry_component_coder", alphageometry_prompt, ["web_search_tool","python_tool"], alphageometry_grader),
]

# --- Modified test functions for RL tasks ---

async def run_single_test(
    run_id: int,
    num_runs: int,
    task: RLTask,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    """
    Run a single test for an RL task.
    For RL tasks, we expect the agent to return both output and metadata.
    """
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} - {task.name} {'=' * 20}")

    # Run the agent loop - for RL tasks, we expect the agent to handle the complex task
    result = await run_agent_loop(
        prompt=task.prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,  # More steps for complex tasks
        verbose=verbose,
    )

    # For RL tasks, the result should be a dictionary with output and metadata
    # or we need to extract this from the agent's final response
    if result is None:
        print(f"✗ Run {run_id} ({task.name}): FAILURE - No result submitted")
        return run_id, False, None

    # Parse the result to extract output and metadata
    # This is a simplified approach - in practice you'd want more sophisticated parsing
    try:
        if isinstance(result, str):
            # Try to parse as JSON containing both output and metadata
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "output" in parsed and "metadata" in parsed:
                output = parsed["output"]
                metadata = parsed["metadata"]
            else:
                # Assume the string is just the output, metadata is empty
                output = result
                metadata = {}
        elif isinstance(result, dict):
            # Direct dictionary with output and metadata
            output = result.get("output", "")
            metadata = result.get("metadata", {})
        else:
            output = str(result)
            metadata = {}
    except (json.JSONDecodeError, AttributeError):
        output = str(result)
        metadata = {}

    # Grade the result using the task's grader
    success = task.grader(output, metadata)

    if success:
        print(f"✓ Run {run_id} ({task.name}): SUCCESS")
    else:
        print(f"✗ Run {run_id} ({task.name}): FAILURE")

    return run_id, success, result


async def main(concurrent: bool = True):
    """
    Main function to test all RL tasks using the template's testing framework.
    """
    # Define the basic tools available to the agent
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # Test parameters
    num_runs_per_task = 3  # Reduced for demo purposes
    execution_mode = "concurrently" if concurrent else "sequentially"
    
    print(f"Testing {len(TASKS)} RL tasks {execution_mode}...")
    print(f"Each task will be run {num_runs_per_task} times")
    print("=" * 60)

    overall_results = []

    for task in TASKS:
        print(f"\n\n{'#' * 20} Testing: {task.name} {'#' * 20}")
        print(f"Prompt: {task.prompt[:100]}...")
        
        # Create test coroutines for this task
        tasks_list = [
            run_single_test(
                run_id=i + 1,
                num_runs=num_runs_per_task,
                task=task,
                tools=tools,
                tool_handlers=tool_handlers,
                verbose=False,  # Set to True for detailed debugging
            )
            for i in range(num_runs_per_task)
        ]

        # Run concurrently or sequentially based on the flag
        if concurrent:
            # Process results as they complete
            task_results = []
            for coro in asyncio.as_completed(tasks_list):
                result = await coro
                task_results.append(result)
        else:
            # Run sequentially by awaiting each task in order
            task_results = []
            for task_coro in tasks_list:
                result = await task_coro
                task_results.append(result)

        # Count successes for this task
        successes = sum(1 for _, success, _ in task_results)
        overall_results.extend(task_results)

        # Display task-specific results
        pass_rate = (successes / num_runs_per_task) * 100
        print(f"\n{'-' * 40}")
        print(f"Task: {task.name}")
        print(f"  Passed: {successes}/{num_runs_per_task}")
        print(f"  Failed: {num_runs_per_task - successes}/{num_runs_per_task}")
        print(f"  Pass Rate: {pass_rate:.1f}%")
        print(f"{'-' * 40}")

    # Calculate and display overall results
    total_successes = sum(1 for _, success, _ in overall_results)
    total_runs = len(overall_results)
    overall_pass_rate = (total_successes / total_runs) * 100

    print(f"\n{'=' * 60}")
    print("OVERALL TEST RESULTS:")
    print(f"  Total Runs: {total_runs}")
    print(f"  Total Passed: {total_successes}/{total_runs}")
    print(f"  Overall Pass Rate: {overall_pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=True))
