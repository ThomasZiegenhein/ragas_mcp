# ragas_mcp

Welcome to **ragas_mcp** – exposing ragas evaluation as MCP server for Agents. We also provide an example for an agent that can pull context and ground truth on it's own and evaluate his answer in a final step. To have more control, we bundle the single metric calls into one mcp call for the agent in this example.

## Features

- **Ragas Single Turn Metric as MCP service**: A MCP server that holds the single turn metrics from ragas. The MCP server can be called by agents for self evaluation. 

## Installation

Clone the repo or copy past the code under mcp_metric_server/server.py

## Quickstart

### 1. Clone the repo

```sh
git clone https://github.com/yourusername/ragas_mcp.git
cd ragas_mcp
```

### 2. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
```
--> If you want to use other LLM-as-a-Judge, change or replace the mcp_metric_server/my_llms.py 

### 3. Build and start all servers

**For running the example**
```sh
docker compose -f docker_composer.yml up --build -d
```

**Running only the MCP metric server**
**For running the example**
```sh
docker compose -f docker_composer_metrics_only.yml up --build -d
```

### 4. Run the agent
**Only if you run the example agent**
```sh
python agent/agent.py
```

## Architecture for the example

```mermaid
flowchart LR
    User["User"]
    Agent["LangChain Agent"]
    subgraph MCP_Metric_Server["MCP RAGAS Metric Server"]
        Faithfulness["Tool: calculate_faithfulness"]
        AnswerRelevancy["Tool: calculate_answer_relevancy"]
        ContextPrecision["Tool: calculate_context_precision"]
        ContextRecall["Tool: calculate_context_recall"]
        AnswerCorrectness["Tool: calculate_answer_correctness"]
    end
    subgraph ContextR["MCP Contect Retrieval Server"]
        CR["Tool: get_context"]
    end
    subgraph EvalWF["MCP Pre-defined Eval Workflows]
        QA["Tool: evaluate_question_answer_with_context_workflow"]
    end
    subgraph ReferenceR["MCP Reference Server"]
        RR["Tool: get_reference_data"]
    end

    User --question--> Agent
    Agent --> EvalWF
    Workflow --> Metric
    Workflow --> Context
    Workflow --> Reference
    Metric <--> Workflow
    Context <--> Workflow
    Reference <--> Workflow
```

## License

MIT

---

**Questions or feedback?**  
Open an issue or reach out via the Medium