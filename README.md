# 🧠 Wumpus World AI Agent (Logic-Based)

This project is a **web-based intelligent agent** that solves the Wumpus World using **Propositional Logic and Resolution Refutation**.

## 🚀 Features

- Knowledge-Based Agent (no hardcoded paths)
- CNF Conversion (Biconditional, Implication elimination)
- Resolution Engine for logical inference
- Dynamic grid environment (Wumpus, pits, gold)
- Real-time visualization:
  - Agent movement
  - Knowledge Base updates
  - CNF / proof traces

## 🧠 How It Works

1. The agent perceives:
   - Breeze → possible pits
   - Stench → possible Wumpus
   - Glitter → gold

2. These percepts are added to a **Knowledge Base**

3. The system:
   - Converts formulas into **CNF**
   - Uses **Resolution Refutation** to prove:
     - Safe cells (`¬Pit ∧ ¬Wumpus`)

4. The agent moves only to **logically proven safe cells**

---

## ⚙️ Tech Stack

- React + TypeScript
- Custom Logic Parser
- Resolution-based Inference Engine

---

## ⚠️ Challenges

### 1. Logic Parser
- Converting complex formulas to CNF
- Handling:
  - Biconditionals (↔)
  - Implications (→)
  - De Morgan’s Laws
  - Distribution rules

### 2. Resolution Engine
- Clause explosion problem
- Avoiding duplicate clauses
- Efficient contradiction detection

### 3. UI Integration
- Syncing logic inference with UI updates
- Visualizing abstract reasoning (CNF, proofs)

---

## 📸 Demo

https://wumpusagent.netlify.app/

---

## 🎯 Learning Outcomes

- Practical understanding of **Propositional Logic**
- CNF transformation & Resolution Refutation
- Building **AI agents from scratch**
- Connecting theory with real-world applications

---

## ▶️ Run Locally

```bash
npm install
npm run dev
