import { useState, useEffect, useCallback, useRef } from "react";

type CellState = "unknown" | "safe" | "visited" | "hazard" | "agent" | "pit" | "wumpus" | "gold";

interface Cell {
  row: number; col: number;
  state: CellState;
  hasPit: boolean; hasWumpus: boolean; hasGold: boolean;
  breeze: boolean; stench: boolean; glitter: boolean;
  visited: boolean;
}

interface CNFClause { literals: string[]; }

interface KBEntry {
  type: "percept" | "safe" | "inference" | "axiom" | "biconditional";
  cell: string; content: string; step: number;
}

interface LogEntry {
  step: number; message: string;
  type: "info" | "infer" | "move" | "warn" | "success" | "percept" | "cnf" | "arrow";
}

interface Metrics {
  totalInferenceSteps: number;
  moves: number;
  percepts: string[];
  safeCellsInferred: number;
  knownHazards: number;
  arrowFired: boolean;
  wumpusKilled: boolean;
  score: number;
}

const cellId = (r: number, c: number) => `${r}_${c}`;
const parseCellId = (id: string) => { const [r, c] = id.split("_").map(Number); return { r, c }; };
const adj = (r: number, c: number, rows: number, cols: number) =>
  [[r-1,c],[r+1,c],[r,c-1],[r,c+1]]
    .filter(([nr,nc]) => nr>=0 && nr<rows && nc>=0 && nc<cols)
    .map(([nr,nc]) => ({r:nr,c:nc}));

const negLit = (lit: string) => lit.startsWith("!") ? lit.slice(1) : `!${lit}`;
const clauseKey = (lits: string[]) => [...lits].sort().join("|");

type Formula =
  | { type: "atom"; name: string }
  | { type: "not"; sub: Formula }
  | { type: "and"; left: Formula; right: Formula }
  | { type: "or"; left: Formula; right: Formula }
  | { type: "impl"; left: Formula; right: Formula }
  | { type: "bicond"; left: Formula; right: Formula };

function atom(name: string): Formula { return { type: "atom", name }; }
function not(sub: Formula): Formula { return { type: "not", sub }; }
function and(left: Formula, right: Formula): Formula { return { type: "and", left, right }; }
function or(left: Formula, right: Formula): Formula { return { type: "or", left, right }; }
function impl(left: Formula, right: Formula): Formula { return { type: "impl", left, right }; }
function bicond(left: Formula, right: Formula): Formula { return { type: "bicond", left, right }; }

function eliminateBicond(f: Formula): Formula {
  switch (f.type) {
    case "atom": return f;
    case "not": return not(eliminateBicond(f.sub));
    case "and": return and(eliminateBicond(f.left), eliminateBicond(f.right));
    case "or": return or(eliminateBicond(f.left), eliminateBicond(f.right));
    case "impl": return impl(eliminateBicond(f.left), eliminateBicond(f.right));
    case "bicond": {
      const l = eliminateBicond(f.left), r = eliminateBicond(f.right);
      return and(impl(l, r), impl(r, l));
    }
  }
}

function eliminateImpl(f: Formula): Formula {
  switch (f.type) {
    case "atom": return f;
    case "not": return not(eliminateImpl(f.sub));
    case "and": return and(eliminateImpl(f.left), eliminateImpl(f.right));
    case "or": return or(eliminateImpl(f.left), eliminateImpl(f.right));
    case "impl": return or(not(eliminateImpl(f.left)), eliminateImpl(f.right));
    case "bicond": return eliminateImpl(eliminateBicond(f));
  }
}

function moveNotInward(f: Formula): Formula {
  switch (f.type) {
    case "atom": return f;
    case "and": return and(moveNotInward(f.left), moveNotInward(f.right));
    case "or": return or(moveNotInward(f.left), moveNotInward(f.right));
    case "impl": return moveNotInward(eliminateImpl(f));
    case "bicond": return moveNotInward(eliminateBicond(f));
    case "not": {
      switch (f.sub.type) {
        case "atom": return f;
        case "not": return moveNotInward(f.sub.sub);
        case "and": return or(moveNotInward(not(f.sub.left)), moveNotInward(not(f.sub.right)));
        case "or": return and(moveNotInward(not(f.sub.left)), moveNotInward(not(f.sub.right)));
        case "impl": return moveNotInward(not(eliminateImpl(f.sub)));
        case "bicond": return moveNotInward(not(eliminateBicond(f.sub)));
      }
    }
  }
}

function distribute(f: Formula): Formula {
  switch (f.type) {
    case "atom": return f;
    case "not": return f;
    case "and": return and(distribute(f.left), distribute(f.right));
    case "or": {
      const l = distribute(f.left), r = distribute(f.right);
      if (l.type === "and") return and(distribute(or(l.left, r)), distribute(or(l.right, r)));
      if (r.type === "and") return and(distribute(or(l, r.left)), distribute(or(l, r.right)));
      return or(l, r);
    }
    default: return f;
  }
}

function extractClauses(f: Formula): string[][] {
  if (f.type === "and") return [...extractClauses(f.left), ...extractClauses(f.right)];
  return [extractLiterals(f)];
}

function extractLiterals(f: Formula): string[] {
  if (f.type === "atom") return [f.name];
  if (f.type === "not" && f.sub.type === "atom") return [`!${f.sub.name}`];
  if (f.type === "or") return [...extractLiterals(f.left), ...extractLiterals(f.right)];
  return [];
}

function toCNF(formula: Formula): string[][] {
  return extractClauses(distribute(moveNotInward(eliminateImpl(eliminateBicond(formula)))));
}

class ResolutionEngine {
  private clauses: CNFClause[] = [];
  private stepCount = 0;
  private lastProofLog: string[] = [];

  addFormula(formula: Formula) {
    const cnfClauses = toCNF(formula);
    for (const lits of cnfClauses) {
      this.clauses.push({ literals: [...new Set(lits)] });
    }
  }

  addClause(lits: string[]) {
    this.clauses.push({ literals: [...new Set(lits)] });
  }

  getStepCount() { return this.stepCount; }
  resetSteps() { this.stepCount = 0; this.lastProofLog = []; }
  getProofLog() { return [...this.lastProofLog]; }

  prove(query: Formula): boolean {
    this.resetSteps();
    const negQueryCNF = toCNF(not(query));
    const workingClauses: CNFClause[] = [
      ...this.clauses.map(c => ({ literals: [...c.literals] })),
      ...negQueryCNF.map(lits => ({ literals: [...new Set(lits)] }))
    ];

    this.lastProofLog.push(`KB: ${this.clauses.length} clauses. Negated query: ${negQueryCNF.map(c => `{${c.join(",")}}`).join(" ^ ")}`);

    const seen = new Set<string>();
    workingClauses.forEach(c => seen.add(clauseKey(c.literals)));

    let changed = true;
    while (changed) {
      changed = false;
      const n = workingClauses.length;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const resolvents = this.resolveTwo(workingClauses[i], workingClauses[j]);
          this.stepCount++;
          for (const resolvent of resolvents) {
            if (resolvent.literals.length === 0) {
              this.lastProofLog.push(`Empty clause derived after ${this.stepCount} steps - PROVEN`);
              return true;
            }
            const key = clauseKey(resolvent.literals);
            if (!seen.has(key)) {
              seen.add(key);
              workingClauses.push(resolvent);
              changed = true;
            }
          }
        }
      }
    }
    this.lastProofLog.push(`No contradiction after ${this.stepCount} steps - UNPROVABLE`);
    return false;
  }

  private resolveTwo(c1: CNFClause, c2: CNFClause): CNFClause[] {
    const results: CNFClause[] = [];
    for (const lit of c1.literals) {
      const comp = negLit(lit);
      if (c2.literals.includes(comp)) {
        const merged = [
          ...c1.literals.filter(l => l !== lit),
          ...c2.literals.filter(l => l !== comp)
        ];
        const unique = [...new Set(merged)];
        const isTautology = unique.some(l => unique.includes(negLit(l)));
        if (!isTautology) results.push({ literals: unique });
      }
    }
    return results;
  }

  clone(): ResolutionEngine {
    const eng = new ResolutionEngine();
    eng.clauses = this.clauses.map(c => ({ literals: [...c.literals] }));
    return eng;
  }
}

class KnowledgeBase {
  private engine: ResolutionEngine = new ResolutionEngine();
  private entries: KBEntry[] = [];
  private uiSafeHints = new Set<string>();
  private uiPitHints = new Set<string>();
  private uiWumpusHints = new Set<string>();
  private wumpusAlive = true;
  private wumpusLocated: string | null = null;
  private entryStep = 0;
  private rows: number;
  private cols: number;

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.engine.addClause([`!P_0_0`]);
    this.engine.addClause([`!W_0_0`]);
    this.uiSafeHints.add("0_0");
    this.addEntry("axiom", "0_0", "Start cell (0,0) is safe: !P_0_0 ^ !W_0_0");
  }

  private addEntry(type: KBEntry["type"], cell: string, content: string) {
    this.entryStep++;
    this.entries.push({ type, cell, content, step: this.entryStep });
  }

  tell(r: number, c: number, breeze: boolean, stench: boolean): string[] {
    const cid = cellId(r, c);
    const neighbors = adj(r, c, this.rows, this.cols);
    const cnfAdded: string[] = [];

    this.engine.addFormula(not(atom(`P_${cid}`)));
    this.engine.addFormula(not(atom(`W_${cid}`)));
    this.uiSafeHints.add(cid);

    if (!breeze) {
      this.engine.addFormula(not(atom(`B_${cid}`)));
      for (const nb of neighbors) {
        const nid = cellId(nb.r, nb.c);
        this.engine.addFormula(not(atom(`P_${nid}`)));
        if (!this.uiPitHints.has(nid)) this.uiSafeHints.add(nid);
      }
      this.addEntry("percept", cid, `!B_${cid} - no adjacent pits`);
    } else {
      const pitAtoms = neighbors.map(nb => atom(`P_${cellId(nb.r, nb.c)}`));
      const pitDisj = pitAtoms.reduce((acc, p) => acc ? or(acc, p) : p) as Formula;
      this.engine.addFormula(bicond(atom(`B_${cid}`), pitDisj));
      this.engine.addFormula(atom(`B_${cid}`));
      const pStr = neighbors.map(nb => `P_${cellId(nb.r, nb.c)}`).join(" | ");
      this.addEntry("biconditional", cid, `B_${cid} <-> (${pStr})`);
      cnfAdded.push(`B_${cid}`, pStr);
    }

    if (!stench) {
      this.engine.addFormula(not(atom(`S_${cid}`)));
      for (const nb of neighbors) {
        const nid = cellId(nb.r, nb.c);
        this.engine.addFormula(not(atom(`W_${nid}`)));
        if (!this.uiWumpusHints.has(nid)) this.uiSafeHints.add(nid);
      }
      this.addEntry("percept", cid, `!S_${cid} - no adjacent wumpus`);
    } else {
      const wAtoms = neighbors.map(nb => atom(`W_${cellId(nb.r, nb.c)}`));
      const wDisj = wAtoms.reduce((acc, w) => acc ? or(acc, w) : w) as Formula;
      this.engine.addFormula(bicond(atom(`S_${cid}`), wDisj));
      this.engine.addFormula(atom(`S_${cid}`));
      const wStr = neighbors.map(nb => `W_${cellId(nb.r, nb.c)}`).join(" | ");
      this.addEntry("biconditional", cid, `S_${cid} <-> (${wStr})`);
      cnfAdded.push(`S_${cid}`, wStr);
    }

    return cnfAdded;
  }

  askSafe(r: number, c: number): { safe: boolean; noPit: boolean; noWumpus: boolean; steps: number; proofLog: string[] } {
    const cid = cellId(r, c);
    const eng = this.engine.clone();

    eng.resetSteps();
    const noPit = eng.prove(not(atom(`P_${cid}`)));
    const pitSteps = eng.getStepCount();
    const pitLog = eng.getProofLog();

    eng.resetSteps();
    const noWumpus = eng.prove(not(atom(`W_${cid}`)));
    const wSteps = eng.getStepCount();
    const wLog = eng.getProofLog();

    const totalSteps = pitSteps + wSteps;
    const safe = noPit && noWumpus;

    if (safe) {
      this.uiSafeHints.add(cid);
      this.addEntry("inference", cid, `Resolution |- !P_${cid} ^ !W_${cid} (${totalSteps} steps) - SAFE`);
    }

    return { safe, noPit, noWumpus, steps: totalSteps, proofLog: [...pitLog, ...wLog] };
  }

  killWumpus(wid: string) {
    this.wumpusAlive = false;
    this.wumpusLocated = null;
    this.engine.addFormula(not(atom(`W_${wid}`)));
    const { r, c } = parseCellId(wid);
    const neighbors = adj(r, c, this.rows, this.cols);
    for (const nb of neighbors) {
      const nid = cellId(nb.r, nb.c);
      if (!this.uiPitHints.has(nid)) this.uiSafeHints.add(nid);
    }
    this.uiWumpusHints.add(wid);
    this.uiSafeHints.delete(wid);
    this.addEntry("inference", wid, `Wumpus at ${wid} killed - !W_${wid} added to KB`);
  }

  markPit(cid: string) {
    this.uiPitHints.add(cid);
    this.uiSafeHints.delete(cid);
    this.engine.addFormula(atom(`P_${cid}`));
    this.addEntry("inference", cid, `Confirmed pit at ${cid}`);
  }

  isUISafe(cid: string) { return this.uiSafeHints.has(cid); }
  isUIPit(cid: string) { return this.uiPitHints.has(cid); }
  isUIWumpus(cid: string) { return this.uiWumpusHints.has(cid); }
  isWumpusAlive() { return this.wumpusAlive; }
  getWumpusLocation() { return this.wumpusLocated; }
  getEntries() { return [...this.entries]; }
  setWumpusLocated(wid: string) {
    this.wumpusLocated = wid;
    this.uiWumpusHints.add(wid);
    this.engine.addFormula(atom(`W_${wid}`));
    this.uiSafeHints.delete(wid);
    this.addEntry("inference", wid, `Wumpus location asserted into KB via stench reasoning`);
  }
}

function generateWorld(rows: number, cols: number, pitProb = 0.15): Cell[][] {
  const grid: Cell[][] = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => ({
      row: r, col: c, state: "unknown" as CellState,
      hasPit: false, hasWumpus: false, hasGold: false,
      breeze: false, stench: false, glitter: false,
      visited: false,
    }))
  );

  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      if (!(r === 0 && c === 0) && Math.random() < pitProb) grid[r][c].hasPit = true;

  let wr: number, wc: number;
  do { wr = Math.floor(Math.random() * rows); wc = Math.floor(Math.random() * cols); }
  while (wr === 0 && wc === 0);
  grid[wr][wc].hasWumpus = true;

  let gr: number, gc: number;
  do { gr = Math.floor(Math.random() * rows); gc = Math.floor(Math.random() * cols); }
  while (gr === 0 && gc === 0);
  grid[gr][gc].hasGold = true;

  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const neighbors = adj(r, c, rows, cols);
      grid[r][c].breeze = neighbors.some(n => grid[n.r][n.c].hasPit);
      grid[r][c].stench = neighbors.some(n => grid[n.r][n.c].hasWumpus);
      grid[r][c].glitter = grid[r][c].hasGold;
    }

  return grid;
}

function findNextMove(
  grid: Cell[][], kb: KnowledgeBase,
  startR: number, startC: number,
  rows: number, cols: number
): { r: number; c: number } | null {
  const immediateNeighbors = adj(startR, startC, rows, cols);
  for (const nb of immediateNeighbors) {
    const nid = cellId(nb.r, nb.c);
    if (!grid[nb.r][nb.c].visited && kb.isUISafe(nid)) return nb;
  }

  const visited = new Set<string>();
  visited.add(cellId(startR, startC));
  const queue: Array<{ r: number; c: number; firstStep: { r: number; c: number } | null }> = [
    { r: startR, c: startC, firstStep: null }
  ];

  while (queue.length > 0) {
    const curr = queue.shift()!;
    for (const nb of adj(curr.r, curr.c, rows, cols)) {
      const nid = cellId(nb.r, nb.c);
      if (visited.has(nid)) continue;
      visited.add(nid);
      const isSafe = kb.isUISafe(nid);
      const firstStep = curr.firstStep ?? nb;
      if (!grid[nb.r][nb.c].visited && isSafe) return firstStep;
      if (grid[nb.r][nb.c].visited || isSafe) queue.push({ r: nb.r, c: nb.c, firstStep });
    }
  }
  return null;
}

function calculateScore(moves: number, arrowFired: boolean, dead: boolean, goldFound: boolean): number {
  let score = 0;
  if (goldFound) score += 1000;
  if (dead) score -= 1000;
  score -= moves;
  if (arrowFired) score -= 10;
  return score;
}

export default function WumpusAgent() {
  const [rows, setRows] = useState(5);
  const [cols, setCols] = useState(5);
  const [grid, setGrid] = useState<Cell[][]>([]);
  const [agentPos, setAgentPos] = useState({ r: 0, c: 0 });
  const [kb, setKb] = useState<KnowledgeBase | null>(null);
  const [metrics, setMetrics] = useState<Metrics>({
    totalInferenceSteps: 0, moves: 0, percepts: [],
    safeCellsInferred: 0, knownHazards: 0, arrowFired: false, wumpusKilled: false, score: 0
  });
  const [log, setLog] = useState<LogEntry[]>([]);
  const [kbEntries, setKbEntries] = useState<KBEntry[]>([]);
  const [running, setRunning] = useState(false);
  const [gameOver, setGameOver] = useState<null | "dead" | "gold" | "stuck">(null);
  const [speed, setSpeed] = useState(800);
  const [revealAll, setRevealAll] = useState(false);
  const [cnfLog, setCnfLog] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState(0);
  const stepRef = useRef(0);
  const logRef = useRef<HTMLDivElement>(null);
  const runningRef = useRef(false);
  const stateRef = useRef<{ grid: Cell[][], kb: KnowledgeBase, pos: {r:number,c:number}, metrics: Metrics } | null>(null);

  const addLog = useCallback((message: string, type: LogEntry["type"] = "info") => {
    stepRef.current++;
    const step = stepRef.current;
    setLog(prev => [...prev.slice(-120), { step, message, type }]);
    setTimeout(() => logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" }), 50);
  }, []);

  const initWorld = useCallback(() => {
    const newGrid = generateWorld(rows, cols);
    const newKB = new KnowledgeBase(rows, cols);
    stepRef.current = 0;
    runningRef.current = false;
    setGrid(newGrid);
    setKb(newKB);
    setAgentPos({ r: 0, c: 0 });
    setLog([]);
    setKbEntries([]);
    setCnfLog([]);
    setGameOver(null);
    setRunning(false);
    setMetrics({ totalInferenceSteps: 0, moves: 0, percepts: [], safeCellsInferred: 1, knownHazards: 0, arrowFired: false, wumpusKilled: false, score: 0 });
    stateRef.current = null;
    addLog(`World ${rows}x${cols} initialised. Agent at (0,0).`, "info");
    addLog(`KB axiom: !P_0_0 ^ !W_0_0`, "cnf");
  }, [rows, cols, addLog]);

  useEffect(() => { initWorld(); }, []);

  const doStep = useCallback((
    currentGrid: Cell[][], currentKb: KnowledgeBase,
    currentPos: { r: number; c: number }, currentMetrics: Metrics
  ): { grid: Cell[][], kb: KnowledgeBase, pos: { r: number; c: number }, metrics: Metrics, done: boolean } => {
    const { r, c } = currentPos;
    const cell = currentGrid[r][c];
    let newMetrics = { ...currentMetrics };

    if (cell.hasPit) {
      addLog(`Fell into pit at (${r},${c}). Episode over.`, "warn");
      const g = currentGrid.map(row => row.map(cl => ({ ...cl })));
      g[r][c].state = "pit";
      currentKb.markPit(cellId(r, c));
      newMetrics.score = calculateScore(newMetrics.moves, newMetrics.arrowFired, true, false);
      setGrid(g); setMetrics(newMetrics); setKbEntries(currentKb.getEntries()); setGameOver("dead");
      return { grid: g, kb: currentKb, pos: currentPos, metrics: newMetrics, done: true };
    }
    if (cell.hasWumpus && currentKb.isWumpusAlive()) {
      addLog(`Eaten by Wumpus at (${r},${c}). Episode over.`, "warn");
      const g = currentGrid.map(row => row.map(cl => ({ ...cl })));
      g[r][c].state = "wumpus";
      newMetrics.score = calculateScore(newMetrics.moves, newMetrics.arrowFired, true, false);
      setGrid(g); setMetrics(newMetrics); setKbEntries(currentKb.getEntries()); setGameOver("dead");
      return { grid: g, kb: currentKb, pos: currentPos, metrics: newMetrics, done: true };
    }
    if (cell.hasGold) {
      addLog(`Gold found at (${r},${c}). Score +1000.`, "success");
      const g = currentGrid.map(row => row.map(cl => ({ ...cl })));
      g[r][c].visited = true; g[r][c].state = "gold";
      newMetrics.score = calculateScore(newMetrics.moves, newMetrics.arrowFired, false, true);
      setGrid(g); setMetrics(newMetrics); setKbEntries(currentKb.getEntries()); setGameOver("gold");
      return { grid: g, kb: currentKb, pos: currentPos, metrics: newMetrics, done: true };
    }

    const newGrid = currentGrid.map(row => row.map(cl => ({ ...cl })));
    newGrid[r][c].visited = true;
    newGrid[r][c].state = "visited";

    const cnfAdded = currentKb.tell(r, c, cell.breeze, cell.stench);
    const percepts: string[] = [];
    if (cell.breeze) percepts.push("Breeze");
    if (cell.stench) percepts.push("Stench");
    if (cell.glitter) percepts.push("Glitter");
    if (!cell.breeze && !cell.stench && !cell.glitter) percepts.push("None");

    addLog(`Percepts @ (${r},${c}): ${percepts.join(", ")}`, "percept");
    if (cnfAdded.length > 0) addLog(`KB TELL: biconditional CNF added`, "cnf");

    if (cell.stench && currentKb.isWumpusAlive() && !currentKb.getWumpusLocation()) {
      const unvisited = adj(r, c, rows, cols).filter(nb => !newGrid[nb.r][nb.c].visited);
      if (unvisited.length === 1) {
        const wid = cellId(unvisited[0].r, unvisited[0].c);
        currentKb.setWumpusLocated(wid);
        addLog(`Wumpus asserted at (${unvisited[0].r},${unvisited[0].c}) - KB updated`, "infer");
      }
    }

    let arrowFired = newMetrics.arrowFired;
    let wumpusKilled = newMetrics.wumpusKilled;
    const locatedWumpus = currentKb.getWumpusLocation();
    if (locatedWumpus && currentKb.isWumpusAlive() && !arrowFired) {
      const { r: wr, c: wc } = parseCellId(locatedWumpus);
      arrowFired = true; wumpusKilled = true;
      currentKb.killWumpus(locatedWumpus);
      newGrid[wr][wc].hasWumpus = false;
      adj(wr, wc, rows, cols).forEach(nb => { newGrid[nb.r][nb.c].stench = false; });
      addLog(`Arrow fired at (${wr},${wc}) - Wumpus killed. -10 pts.`, "arrow");
    }

    const neighbors = adj(r, c, rows, cols);
    let inferSteps = 0;
    let safeCt = newMetrics.safeCellsInferred;
    let hazardCt = newMetrics.knownHazards;
    const allCnfLogs: string[] = [];

    for (const nb of neighbors) {
      const nid = cellId(nb.r, nb.c);
      if (!newGrid[nb.r][nb.c].visited) {
        const { safe, noPit, noWumpus, steps, proofLog } = currentKb.askSafe(nb.r, nb.c);
        inferSteps += steps;
        allCnfLogs.push(...proofLog);

        addLog(
          `ASK |- !P_${nid} ^ !W_${nid} ? -> ${safe ? "SAFE" : noPit ? "Possible pit" : "Possible wumpus"} [${steps} steps]`,
          "infer"
        );

        if (safe) {
          newGrid[nb.r][nb.c].state = "safe";
          safeCt++;
        } else if (currentKb.isUIPit(nid)) {
          newGrid[nb.r][nb.c].state = "hazard";
          hazardCt++;
        } else if (currentKb.isUIWumpus(nid)) {
          newGrid[nb.r][nb.c].state = "hazard";
          hazardCt++;
        }
      }
    }

    setCnfLog(prev => [...prev.slice(-200), ...allCnfLogs]);

    const next = findNextMove(newGrid, currentKb, r, c, rows, cols);

    newMetrics = {
      ...newMetrics,
      totalInferenceSteps: newMetrics.totalInferenceSteps + inferSteps,
      percepts, safeCellsInferred: safeCt, knownHazards: hazardCt, arrowFired, wumpusKilled,
    };

    if (!next) {
      addLog("No safe moves available - agent halted.", "warn");
      newMetrics.score = calculateScore(newMetrics.moves, arrowFired, false, false);
      setGrid(newGrid); setMetrics(newMetrics); setKbEntries(currentKb.getEntries()); setGameOver("stuck");
      return { grid: newGrid, kb: currentKb, pos: currentPos, metrics: newMetrics, done: true };
    }

    addLog(`Move to (${next.r},${next.c}) - step ${newMetrics.moves + 1}`, "move");
    newMetrics.moves++;

    setGrid(newGrid.map(row => row.map(cl => ({ ...cl })))); setAgentPos(next); setMetrics(newMetrics); setKbEntries(currentKb.getEntries());
    return { grid: newGrid, kb: currentKb, pos: next, metrics: newMetrics, done: false };
  }, [rows, cols, addLog]);

  const runAuto = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    let g = stateRef.current?.grid ?? grid.map(row => row.map(c => ({ ...c })));
    let k = stateRef.current?.kb ?? kb!;
    let pos = stateRef.current?.pos ?? agentPos;
    let m = stateRef.current?.metrics ?? metrics;
    while (runningRef.current) {
      const result = await new Promise<ReturnType<typeof doStep>>(resolve => {
        setTimeout(() => resolve(doStep(g, k, pos, m)), speed);
      });
      g = result.grid; k = result.kb; pos = result.pos; m = result.metrics;
      stateRef.current = { grid: g, kb: k, pos, metrics: m };
      if (result.done || !runningRef.current) break;
    }
    runningRef.current = false;
    setRunning(false);
  }, [grid, kb, agentPos, metrics, doStep, speed]);

  const runOneStep = useCallback(() => {
    const g = stateRef.current?.grid ?? grid.map(row => row.map(c => ({ ...c })));
    const k = stateRef.current?.kb ?? kb!;
    const pos = stateRef.current?.pos ?? agentPos;
    const m = stateRef.current?.metrics ?? metrics;
    const result = doStep(g, k, pos, m);
    stateRef.current = { grid: result.grid, kb: result.kb, pos: result.pos, metrics: result.metrics };
  }, [grid, kb, agentPos, metrics, doStep]);

  const stopAgent = () => { runningRef.current = false; setRunning(false); };

  const getCellDisplay = (cell: Cell, isAgent: boolean) => {
    if (isAgent) return { bg: "#dbeafe", border: "#3b82f6", text: "#1d4ed8", label: "A" };
    if (revealAll) {
      if (cell.hasPit) return { bg: "#fee2e2", border: "#ef4444", text: "#991b1b", label: "P" };
      if (cell.hasWumpus) return { bg: "#fce7f3", border: "#ec4899", text: "#9d174d", label: "W" };
      if (cell.hasGold) return { bg: "#fef9c3", border: "#eab308", text: "#854d0e", label: "G" };
    }
    switch (cell.state) {
      case "visited": return { bg: "#e0f2fe", border: "#7dd3fc", text: "#0369a1", label: "" };
      case "safe":    return { bg: "#dcfce7", border: "#22c55e", text: "#15803d", label: "" };
      case "hazard":  return { bg: "#fff1f2", border: "#fda4af", text: "#9f1239", label: "!" };
      case "gold":    return { bg: "#fef9c3", border: "#eab308", text: "#854d0e", label: "G" };
      case "pit":     return { bg: "#fee2e2", border: "#ef4444", text: "#991b1b", label: "P" };
      case "wumpus":  return { bg: "#fce7f3", border: "#ec4899", text: "#9d174d", label: "W" };
      default:        return { bg: "#f8fafc", border: "#e2e8f0", text: "#94a3b8", label: "" };
    }
  };

  const logColors: Record<LogEntry["type"], string> = {
    info: "#94a3b8", infer: "#3b82f6", move: "#16a34a",
    warn: "#ef4444", success: "#ca8a04", percept: "#7c3aed",
    cnf: "#0891b2", arrow: "#ea580c"
  };

  const cellSize = Math.min(68, Math.floor(Math.min(440 / rows, 480 / cols)));

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", background: "#f8fafc", minHeight: "100vh", color: "#1e293b" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }

        .app { display: flex; flex-direction: column; height: 100vh; overflow: hidden; background: #f8fafc; }

        .hdr {
          background: #ffffff;
          border-bottom: 1px solid #e2e8f0;
          padding: 12px 24px;
          display: flex; align-items: center; gap: 16px;
          flex-shrink: 0;
        }
        .hdr-title { font-family: 'DM Mono', monospace; font-size: 16px; font-weight: 500; color: #0f172a; letter-spacing: 1px; }
        .hdr-sub { font-size: 12px; color: #94a3b8; margin-top: 3px; font-family: 'DM Sans', sans-serif; }
        .hdr-badges { margin-left: auto; display: flex; gap: 6px; }
        .badge { padding: 4px 10px; border-radius: 3px; font-size: 11px; font-weight: 500; letter-spacing: 0.3px; font-family: 'DM Mono', monospace; border: 1px solid; }
        .badge-blue { background: #eff6ff; color: #3b82f6; border-color: #bfdbfe; }
        .badge-green { background: #f0fdf4; color: #16a34a; border-color: #bbf7d0; }
        .badge-slate { background: #f8fafc; color: #64748b; border-color: #e2e8f0; }

        .body { display: flex; flex: 1; overflow: hidden; }

        .panel-left { width: 200px; flex-shrink: 0; background: #ffffff; border-right: 1px solid #e2e8f0; display: flex; flex-direction: column; overflow: hidden; }
        .panel-right { width: 280px; flex-shrink: 0; background: #ffffff; border-left: 1px solid #e2e8f0; display: flex; flex-direction: column; overflow: hidden; }
        .panel-section { border-bottom: 1px solid #f1f5f9; padding: 12px 14px; }
        .panel-label { font-size: 11px; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; font-family: 'DM Sans', sans-serif; font-weight: 600; }

        .center { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #f8fafc; overflow: hidden; position: relative; padding: 12px; }

        .ctrl-row { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }
        .ctrl-input { background: #f8fafc; border: 1px solid #e2e8f0; color: #1e293b; padding: 5px 8px; font-family: 'DM Mono', monospace; font-size: 12px; border-radius: 4px; width: 56px; outline: none; }
        .ctrl-input:focus { border-color: #3b82f6; background: #fff; }
        .ctrl-lbl { font-size: 12px; color: #64748b; min-width: 14px; font-family: 'DM Sans', sans-serif; font-weight: 500; }

        .btn { font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 600; letter-spacing: 0.2px; border: 1px solid; border-radius: 4px; cursor: pointer; padding: 8px 10px; transition: all 0.1s; width: 100%; margin-bottom: 5px; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-primary { background: #eff6ff; color: #3b82f6; border-color: #bfdbfe; }
        .btn-primary:hover:not(:disabled) { background: #dbeafe; }
        .btn-success { background: #f0fdf4; color: #16a34a; border-color: #bbf7d0; }
        .btn-success:hover:not(:disabled) { background: #dcfce7; }
        .btn-danger { background: #fff1f2; color: #ef4444; border-color: #fecdd3; }
        .btn-danger:hover:not(:disabled) { background: #ffe4e6; }
        .btn-ghost { background: #f8fafc; color: #64748b; border-color: #e2e8f0; }
        .btn-ghost:hover:not(:disabled) { background: #f1f5f9; color: #475569; }

        .metric-row { display: flex; flex-direction: column; padding: 9px 14px; border-bottom: 1px solid #f1f5f9; }
        .metric-val { font-family: 'DM Mono', monospace; font-size: 20px; font-weight: 500; line-height: 1; }
        .metric-lbl { font-size: 11px; color: #94a3b8; letter-spacing: 0.3px; margin-top: 3px; text-transform: uppercase; font-family: 'DM Sans', sans-serif; }
        .m-blue { color: #3b82f6; }
        .m-green { color: #16a34a; }
        .m-amber { color: #d97706; }
        .m-red { color: #ef4444; }
        .m-score { color: #0f172a; }

        .percept-area { display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 14px; border-bottom: 1px solid #f1f5f9; }
        .p-tag { padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; font-family: 'DM Sans', sans-serif; border: 1px solid; }
        .p-breeze { background: #eff6ff; color: #3b82f6; border-color: #bfdbfe; }
        .p-stench { background: #fff1f2; color: #ef4444; border-color: #fecdd3; }
        .p-glitter { background: #fefce8; color: #ca8a04; border-color: #fde68a; }
        .p-none { background: #f8fafc; color: #94a3b8; border-color: #e2e8f0; }

        .grid { display: grid; gap: 2px; }
        .cell { display: flex; align-items: center; justify-content: center; border-radius: 4px; border: 1px solid; position: relative; transition: background 0.15s, border-color 0.15s; flex-direction: column; cursor: default; }
        .cell-coord { position: absolute; top: 2px; left: 3px; font-size: 8px; color: #cbd5e1; font-family: 'DM Mono', monospace; line-height: 1; }
        .cell-label { font-size: clamp(11px, 2vw, 16px); font-weight: 600; font-family: 'DM Mono', monospace; }
        .cell-percept { position: absolute; bottom: 2px; right: 3px; font-size: 8px; color: #64748b; font-family: 'DM Mono', monospace; font-weight: 500; }

        .legend { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 16px; border-top: 1px solid #e2e8f0; background: #ffffff; width: 100%; justify-content: center; }
        .legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #64748b; font-family: 'DM Sans', sans-serif; }
        .legend-dot { width: 10px; height: 10px; border-radius: 2px; border: 1px solid; }

        .tab-row { display: flex; border-bottom: 1px solid #e2e8f0; flex-shrink: 0; }
        .tab { flex: 1; padding: 9px 4px; font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 600; letter-spacing: 0.3px; background: transparent; border: none; cursor: pointer; color: #94a3b8; text-transform: uppercase; transition: all 0.1s; border-bottom: 2px solid transparent; }
        .tab.active { color: #3b82f6; border-bottom-color: #3b82f6; }

        .log-scroll { flex: 1; overflow-y: auto; padding: 6px; scrollbar-width: thin; scrollbar-color: #e2e8f0 transparent; }
        .log-entry { padding: 4px 8px; margin-bottom: 1px; border-radius: 3px; font-size: 11px; line-height: 1.55; border-left: 2px solid; display: flex; gap: 6px; align-items: flex-start; background: #f8fafc; }
        .log-step { color: #cbd5e1; font-size: 10px; min-width: 24px; flex-shrink: 0; font-family: 'DM Mono', monospace; }

        .kb-scroll { flex: 1; overflow-y: auto; padding: 6px; scrollbar-width: thin; scrollbar-color: #e2e8f0 transparent; }
        .kb-entry { padding: 4px 8px; margin-bottom: 2px; border-radius: 3px; font-size: 10px; line-height: 1.6; font-family: 'DM Mono', monospace; background: #f8fafc; border: 1px solid #f1f5f9; }
        .kb-tag { color: #3b82f6; font-weight: 600; }
        .kb-formula { color: #475569; }
        .kb-type-axiom { border-left: 2px solid #d97706; }
        .kb-type-biconditional { border-left: 2px solid #3b82f6; }
        .kb-type-inference { border-left: 2px solid #16a34a; }
        .kb-type-percept { border-left: 2px solid #7c3aed; }

        .cnf-scroll { flex: 1; overflow-y: auto; padding: 6px; scrollbar-width: thin; scrollbar-color: #e2e8f0 transparent; }
        .cnf-entry { padding: 3px 8px; margin-bottom: 1px; font-size: 10px; line-height: 1.55; font-family: 'DM Mono', monospace; color: #64748b; border-left: 2px solid #e2e8f0; background: #f8fafc; border-radius: 0 3px 3px 0; }

        .game-over { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 28px 36px; text-align: center; z-index: 10; box-shadow: 0 4px 24px rgba(0,0,0,0.08); min-width: 240px; }
        .go-title { font-family: 'DM Mono', monospace; font-size: 20px; font-weight: 500; letter-spacing: 1px; margin-bottom: 4px; }
        .go-sub { font-size: 13px; color: #64748b; margin-bottom: 14px; font-family: 'DM Sans', sans-serif; }
        .go-stats { font-size: 12px; color: #94a3b8; margin-bottom: 16px; font-family: 'DM Mono', monospace; line-height: 1.8; }

        .slider-row { display: flex; align-items: center; gap: 8px; }
        .slider { -webkit-appearance:none; width:100%; height:2px; border-radius:1px; background:#e2e8f0; outline:none; }
        .slider::-webkit-slider-thumb { -webkit-appearance:none; width:10px; height:10px; border-radius:50%; background:#3b82f6; cursor:pointer; }
        .slider-val { font-size: 11px; color: #3b82f6; min-width: 38px; text-align: right; font-family: 'DM Mono', monospace; }

        .pos-badge { display: flex; align-items: center; gap: 6px; padding: 8px 14px; border-bottom: 1px solid #f1f5f9; }
        .pos-dot { width: 7px; height: 7px; border-radius: 50%; background: #3b82f6; }
        .pos-text { font-family: 'DM Mono', monospace; font-size: 13px; color: #1e293b; font-weight: 500; }
        .pos-lbl { font-size: 11px; color: #94a3b8; font-family: 'DM Sans', sans-serif; }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 2px; }
      `}</style>

      <div className="app">
        <div className="hdr">
          <div>
            <div className="hdr-title">WUMPUS AGENT</div>
            <div className="hdr-sub">Propositional Logic · CNF Conversion · Resolution Refutation</div>
          </div>
          <div className="hdr-badges">
            <span className="badge badge-blue">CNF</span>
            <span className="badge badge-green">RESOLUTION</span>
            <span className="badge badge-slate">BFS</span>
          </div>
        </div>

        <div className="body">
          <div className="panel-left">
            <div className="panel-section">
              <div className="panel-label">Grid Size</div>
              <div className="ctrl-row">
                <span className="ctrl-lbl">R</span>
                <input className="ctrl-input" type="number" min={3} max={10} value={rows}
                  onChange={e => setRows(Math.min(10, Math.max(3, +e.target.value)))} disabled={running} />
                <span className="ctrl-lbl">C</span>
                <input className="ctrl-input" type="number" min={3} max={10} value={cols}
                  onChange={e => setCols(Math.min(10, Math.max(3, +e.target.value)))} disabled={running} />
              </div>
              <button className="btn btn-ghost" onClick={initWorld} disabled={running}>Reset World</button>
            </div>

            <div className="panel-section">
              <div className="panel-label">Step Speed</div>
              <div className="slider-row">
                <input className="slider" type="range" min={100} max={2000} step={100}
                  value={speed} onChange={e => setSpeed(+e.target.value)} />
                <span className="slider-val">{speed}ms</span>
              </div>
            </div>

            <div className="panel-section">
              {!running
                ? <button className="btn btn-primary" onClick={runAuto} disabled={!!gameOver}>Run Auto</button>
                : <button className="btn btn-danger" onClick={stopAgent}>Stop</button>
              }
              <button className="btn btn-success" onClick={runOneStep} disabled={running || !!gameOver}>Step Once</button>
              <button className="btn btn-ghost" onClick={() => setRevealAll(v => !v)}>
                {revealAll ? "Hide World" : "Reveal All"}
              </button>
            </div>

            <div className="metric-row">
              <div className="metric-val m-blue">{metrics.totalInferenceSteps.toLocaleString()}</div>
              <div className="metric-lbl">Inference Steps</div>
            </div>
            <div className="metric-row">
              <div className="metric-val m-green">{metrics.moves}</div>
              <div className="metric-lbl">Agent Moves</div>
            </div>
            <div className="metric-row">
              <div className="metric-val m-amber">{metrics.safeCellsInferred}</div>
              <div className="metric-lbl">Safe Cells (Proven)</div>
            </div>
            <div className="metric-row">
              <div className="metric-val m-score" style={{ color: metrics.score >= 0 ? "#16a34a" : "#ef4444" }}>
                {metrics.score > 0 ? "+" : ""}{metrics.score}
              </div>
              <div className="metric-lbl">Score</div>
            </div>
            <div className="metric-row">
              <div className="metric-val m-red">{metrics.knownHazards}</div>
              <div className="metric-lbl">Hazards Detected</div>
            </div>

            <div className="pos-badge">
              <div className="pos-dot" />
              <span className="pos-text">({agentPos.r},{agentPos.c})</span>
              <span className="pos-lbl">position</span>
            </div>

            {metrics.arrowFired && (
              <div style={{ padding: "6px 14px", fontSize: 11, color: "#ea580c", fontFamily: "'DM Sans', sans-serif", borderBottom: "1px solid #f1f5f9" }}>
                Arrow fired {metrics.wumpusKilled ? "- Wumpus killed" : "- Miss"}
              </div>
            )}

            <div className="percept-area">
              {metrics.percepts.length === 0
                ? <span style={{ fontSize: 12, color: "#cbd5e1", fontFamily: "'DM Sans', sans-serif" }}>-</span>
                : metrics.percepts.map(p => (
                  <span key={p} className={`p-tag p-${p.toLowerCase()}`}>{p}</span>
                ))
              }
            </div>
          </div>

          <div className="center">
            {grid.length > 0 && (
              <div className="grid" style={{
                gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
                gridTemplateRows: `repeat(${rows}, ${cellSize}px)`
              }}>
                {grid.map((row, r) => row.map((cell, c) => {
                  const isAgent = agentPos.r === r && agentPos.c === c;
                  const display = getCellDisplay(cell, isAgent);
                  return (
                    <div key={`${r}-${c}`}
                      className="cell"
                      style={{ width: cellSize, height: cellSize, background: display.bg, borderColor: display.border }}>
                      <span className="cell-coord">{r},{c}</span>
                      <span className="cell-label" style={{ color: display.text }}>{display.label}</span>
                      {cell.visited && !isAgent && (
                        <span className="cell-percept">
                          {cell.breeze ? "B" : ""}{cell.stench ? "S" : ""}
                        </span>
                      )}
                    </div>
                  );
                }))}
              </div>
            )}

            <div className="legend">
              {[
                { bg: "#dbeafe", border: "#3b82f6", label: "Agent" },
                { bg: "#e0f2fe", border: "#7dd3fc", label: "Visited" },
                { bg: "#dcfce7", border: "#22c55e", label: "Safe (proven)" },
                { bg: "#fff1f2", border: "#fda4af", label: "Hazard" },
                { bg: "#f8fafc", border: "#e2e8f0", label: "Unknown" },
              ].map(item => (
                <div className="legend-item" key={item.label}>
                  <div className="legend-dot" style={{ background: item.bg, borderColor: item.border }} />
                  <span>{item.label}</span>
                </div>
              ))}
            </div>

            {gameOver && (
              <div className="game-over" style={{
                borderColor: gameOver === "gold" ? "#eab308" : gameOver === "stuck" ? "#3b82f6" : "#ef4444"
              }}>
                <div className="go-title" style={{
                  color: gameOver === "gold" ? "#ca8a04" : gameOver === "stuck" ? "#3b82f6" : "#ef4444"
                }}>
                  {gameOver === "gold" ? "GOLD FOUND" : gameOver === "dead" ? "AGENT DEAD" : "HALTED"}
                </div>
                <div className="go-sub">
                  {gameOver === "gold" ? "Mission accomplished." : gameOver === "dead" ? "Agent entered a hazardous cell." : "No reachable safe unvisited cells."}
                </div>
                <div className="go-stats">
                  {metrics.moves} moves · {metrics.totalInferenceSteps.toLocaleString()} inference steps<br />
                  Score: {metrics.score > 0 ? "+" : ""}{metrics.score}
                </div>
                <button className="btn btn-primary" onClick={initWorld} style={{ marginTop: 4 }}>New Episode</button>
              </div>
            )}
          </div>

          <div className="panel-right">
            <div className="tab-row">
              {["Log", "KB", "CNF Trace"].map((t, i) => (
                <button key={t} className={`tab ${activeTab === i ? "active" : ""}`} onClick={() => setActiveTab(i)}>{t}</button>
              ))}
            </div>

            {activeTab === 0 && (
              <div className="log-scroll" ref={logRef}>
                {log.map((entry, i) => (
                  <div key={i} className="log-entry" style={{ borderLeftColor: logColors[entry.type] }}>
                    <span className="log-step">#{entry.step}</span>
                    <span style={{ color: logColors[entry.type], fontSize: 11 }}>{entry.message}</span>
                  </div>
                ))}
                {log.length === 0 && (
                  <div style={{ padding: 16, textAlign: "center", color: "#cbd5e1", fontSize: 12, fontFamily: "'DM Sans', sans-serif" }}>
                    Press Run or Step to begin
                  </div>
                )}
              </div>
            )}

            {activeTab === 1 && (
              <div className="kb-scroll">
                <div style={{ padding: "5px 7px 7px", fontSize: 11, color: "#94a3b8", borderBottom: "1px solid #f1f5f9", marginBottom: 4, fontFamily: "'DM Mono', monospace" }}>
                  {kbEntries.length} entries
                </div>
                {kbEntries.map((entry, i) => (
                  <div key={i} className={`kb-entry kb-type-${entry.type}`}>
                    <span className="kb-tag">#{entry.step} [{entry.type}] ({entry.cell.replace("_", ",")}) </span>
                    <span className="kb-formula">{entry.content}</span>
                  </div>
                ))}
                {kbEntries.length === 0 && (
                  <div style={{ padding: 16, textAlign: "center", color: "#cbd5e1", fontSize: 12, fontFamily: "'DM Sans', sans-serif" }}>
                    KB empty
                  </div>
                )}
              </div>
            )}

            {activeTab === 2 && (
              <div className="cnf-scroll">
                <div style={{ padding: "5px 7px 7px", fontSize: 11, color: "#94a3b8", borderBottom: "1px solid #f1f5f9", marginBottom: 4, fontFamily: "'DM Mono', monospace" }}>
                  Resolution proof trace
                </div>
                {cnfLog.map((line, i) => (
                  <div key={i} className="cnf-entry">{line}</div>
                ))}
                {cnfLog.length === 0 && (
                  <div style={{ padding: 16, textAlign: "center", color: "#cbd5e1", fontSize: 12, fontFamily: "'DM Sans', sans-serif" }}>
                    Proof trace will appear here
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
