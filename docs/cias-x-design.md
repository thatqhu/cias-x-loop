## 1. System Overview

### 1.1 Purpose

CIAS-X is an autonomous AI scientist system designed to iteratively discover optimal configurations for Snapshot Compressive Imaging (SCI) reconstruction tasks. The system automates the cycle of:

- **Planning**: Proposing new experimental configurations

- **Execution**: Running experiments and collecting metrics and artifacts

- **Analysis**: Computing Pareto frontiers and summarizing trends

### 1.2 Datebase

Below is thew world model design. We will use sqlite3 as database.



![Untitled Diagram.drawio (2).png](./cias-x.png)


The `pareto_frontiers` table will save top 10 pareto frontier for each strata in current design scope.
We have 3 tiers data structure in world mode:
**Tier1**: The meta data: configs, metrics and artifacs. --> executor generated
**Tier2**: The sumarized info: plan summary. --> analyst generated
**Tier3**: The global_summary and global pareto frontier. --> planer used

### 1.2 Workflow

This project will use langgraph as the agent workflow handler. The graph state should contains values like these:

    class AgentState(TypedDict):
        """
        Global State for the AI Scientist Multi-Agent Workflow (LangGraph)
        """
        executed_experiment_count: int

        # Static / Read-only
        design_space: Dict[str, Any]
        budget_remaining: int


        # Design info
        design_id: int

        # Planner
        configs: List[Any]  # Proposed configs for current plan cycle

        #Executor
        experiments: List[Any] # Experiment results for current plan cycle

        # Analyst
        pareto_frontiers: List[Any] # Pareto frontiers from WorldModel
        global_summary: str # Global summary from WorldModel

        # Workflow Control
        status: str  # "planning", "executing", "analyzing", "end"

planner -> Executor -> analyst

The **planner Agent**: check `design_id` in state. If there is no such id, create a new design record and set the design id.
The planner then use below logic to generate new configs:

    1:  gaps      ← IDENTIFY_UNDEREXPLORED_REGIONS(global_summary)
    2:  frontiers ← pareto_frontiers

    3:  // Prepare textual prompt for LLM planner
    4:  prompt ← BUILD_PLANNER_PROMPT(gaps, frontiers)

    5:  // Invoke LLM to propose configs (in a controlled schema)
    6:  proposals_raw ← LLM_GENERATE_CONFIGS(prompt)

    7: // Validate and project proposals into design space 𝒮
    8: C_new ← ∅
    9: for each p in proposals_raw do
    10:     c' = PROJECT_TO_DESIGN_SPACE(p, 𝒮)
    15:     if IS_VALID_CONFIG(c', constraints) then
    16:         C_new ← C_new ∪ {c'}
    17:     end if
    18: end for

For the first time(there is no global_summary, frontiers and experiments), let LLM give a baseline config.

The planner use the top k pareto frontiers and the `global_summary` to generate the next new configs.

The new config should be saved in `configs` in graph state.

The **Executor Agent** can be blank as I already has an implementation. Currently, just do these things:

1. Create a new record in `plans` table.

2. Save results to `experiments` table(configs, metrics and artifacts).

3. Update `experiments` in graph state.

The **Analyst Agent** will combine current experiments and the experiments in `pareto_frontiers` table to get the new top k pareto frotiers for each strata, then save them back to `pareto_frontiers` table. the description for `pareto_frontiers` table:

-- `experiment_id` - `experiments`.experiment_id
-- `rank` - the pareto frontier rank
-- `strata` - 'T', 'dose'...

This agent will also grab current experiments and pareto frontiers in prompt. Then pass this prompt to LLM to get a plan summary, the summary should follow below rules:
1. about 3-6 sentences in total
1. summarize current experiments in plan scope
2. contains `recommandation`
3. contains `trends`

After updated the summary in `plans` table, this agent will then check whether there are 50 plans executed after last time we calculate the global_summary(`designs`.`last_summary_plan_id` should be the new baseline for global_summary). If it is true, update the global_summary with old global_summary + current 50 summaries with 5-10 sentences in total.

The algorithm for pareto frontier:

    // Group by relevant strata (e.g., (T, dose, mask family))
    4:  strata ← GROUP_BY_STRATA(E_all)

    5:  P ← ∅
    6:  trends ← ∅

    7:  for each s in strata do
    8:      E_s ← strata[s]
    9:
    10:     // Build multi-objective frontiers, e.g. PSNR vs coverage vs latency
    11:     P_s = COMPUTE_PARETO_FRONT(E_s, objectives = {PSNR, coverage, latency})
    12:     P ← P ∪ P_s
    13:
    14:     // Compute calibration diagnostics
    15:     calib_stats_s = COMPUTE_CALIBRATION_STATS(E_s)
    16:
    17:     // Summarize design patterns in natural language
    18:     t_s = SUMMARIZE_TRENDS(E_s, P_s, calib_stats_s)
    19:     trends ← trends ∪ {t_s}
    20:  end for
