# Dexter 🤖

Dexter is an autonomous long-only trading agent that thinks, plans, and iterates as it builds conviction. It plans tasks, inspects quantitative edges, and produces risk-aware playbooks designed to maximize upside while protecting capital.


<img width="979" height="651" alt="Screenshot 2025-10-14 at 6 12 35 PM" src="https://github.com/user-attachments/assets/5a2859d4-53cf-4638-998a-15cef3c98038" />

## Overview

Dexter turns ambitious return targets into sequenced trading workflows. It pulls market data from free sources, validates fundamentals, quantifies technical momentum, and returns a ready-to-execute long thesis with position sizing and risk notes.

**Key Capabilities:**
- **Task Planning**: Breaks complex trading prompts into targeted data pulls and analyses.
- **Market Data Access**: Uses Yahoo Finance (no paid key needed) for prices and fundamentals.
- **Long Strategy Analytics**: Evaluates moving-average crossover performance versus benchmarks.
- **Risk Management Guidance**: Surfaces volatility, drawdown, and monitoring triggers.
- **Safety Features**: Built-in loop detection and step caps prevent runaway execution.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- xAI API key with access to Grok 4 (get [here](https://docs.x.ai/docs/overview))  
  (Optional: set `DEXTER_LLM_PROVIDER=openai` to use OpenAI instead)
- No paid market data key required (Yahoo Finance powers the trading tools)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/virattt/dexter.git
cd dexter
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Set up your environment variables:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your API keys
# DEXTER_LLM_PROVIDER=xai
# XAI_API_KEY=your-xai-api-key
# XAI_MODEL=grok-4-fast-reasoning
# (Optional) configure the OPENAI_* variables if you want to use OpenAI instead
# No market data key required (Yahoo Finance)
```

### Usage

Run Dexter in interactive mode:
```bash
uv run dexter-agent
```

### Example Queries

Try asking Dexter questions like:
- "Build a long plan for NVDA that maximizes risk-adjusted returns."
- "Is MSFT still a buy if I'm targeting 15% annualized with limited drawdowns?"
- "Compare AAPL versus SPY and tell me if now is a good long entry."
- "Identify the best entry plan for META with moving average confirmation."

Dexter will automatically:
1. Break your request into market data, fundamentals, and strategy evaluation tasks.
2. Fetch prices and fundamentals from Yahoo Finance.
3. Quantify trend strength, returns, drawdown, and benchmark advantage.
4. Produce an actionable long recommendation with position management guidance.

## Architecture

Dexter uses a multi-agent architecture with specialized components:

- **Planning Agent**: Breaks a trading question into sequenced data pulls.
- **Action Agent**: Chooses the right tool (prices, fundamentals, strategy eval) for each step.
- **Validation Agent**: Confirms when tasks have enough evidence to proceed.
- **Answer Agent**: Synthesizes the final long recommendation, risks, and execution plan.

## Project Structure

```
dexter/
├── src/
│   ├── dexter/
│   │   ├── agent.py      # Main agent orchestration logic
│   │   ├── model.py              # LLM interface (Grok 4 by default)
│   │   ├── prompts.py            # System prompts for each component
│   │   ├── schemas.py            # Pydantic models
│   │   ├── tools/
│   │   │   ├── market_data.py    # Yahoo Finance OHLCV access
│   │   │   ├── fundamentals.py   # Fundamental snapshot helper
│   │   │   └── long_strategy.py  # Long-only strategy evaluation
│   │   ├── utils/                # UI + logging helpers
│   │   └── cli.py                # CLI entry point
├── pyproject.toml
└── uv.lock
```

## Configuration

Dexter supports configuration via the `Agent` class initialization:

```python
from dexter.agent import Agent

agent = Agent(
    max_steps=20,              # Global safety limit
    max_steps_per_task=5       # Per-task iteration limit
)
```

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.


## License

This project is licensed under the MIT License.
