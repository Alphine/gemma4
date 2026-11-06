# Paired LoRA eval `lora-ab1`

- base arm `edit-early-v4` vs adapter arm `rft-pilot1-sft-full` (`rft_lora` sha256 a3d7ba481eb3); budget {'max_time_minutes': 5.5, 'max_tool_calls': 60, 'max_turns': 120, 'timeout_seconds': 300}; concurrency 8; samples 1
- tasks in order 144, with a complete pair 144, pairs 144
- **no significant difference: +1.4% (95% CI -4.2%..+6.9%, McNemar p=0.804, 144 pairs)**

## Resolved (paired)

| | adapter resolved | adapter not resolved |
|---|---|---|
| **base resolved** | 63 | 7 |
| **base not resolved** | 9 | 65 |

base 70/144 = 0.4861, adapter 72/144 = 0.5, diff 0.0139 (95% bootstrap CI [-0.0417, 0.0694]), exact McNemar p = 0.803619

## Per arm (all finished runs)

| metric | base | adapter |
|---|---|---|
| units (incl. infra) | 144 | 144 |
| infra errors | 0 | 0 |
| finished | 144 | 144 |
| resolved | 70 | 72 |
| resolve rate | 0.4861 | 0.5 |
| infra rate | 0.0 | 0.0 |
| timeouts | 59 | 57 |
| timeout rate | 0.4097 | 0.3958 |
| empty patches | 12 | 9 |
| empty-patch rate | 0.0833 | 0.0625 |
| mean agent min | 3.847 | 3.985 |
| mean tool calls | 29.667 | 30.146 |
| mean LLM calls | 31.694 | 31.799 |
| mean LLM latency s | 7.212 | 7.446 |
| mean runs in flight at start | 7.806 | 7.799 |
| mean patch chars | 846.917 | 906.174 |
| skill calls | 277 | 198 |
| skill calls malformed | 30 | 25 |
| skill malformed rate | 0.1083 | 0.1263 |
| all tool calls | 4473 | 4484 |
| all malformed | 30 | 25 |
| all malformed rate | 0.0067 | 0.0056 |

error types: base {'Agent exceeded session timeout (5.5 min)': 54}, adapter {'Agent exceeded session timeout (5.5 min)': 52}

## Cost

`{'spent_usd': 15.104, 'units_total': 288, 'units_finished': 288, 'usd_per_finished_run_marginal': 0.0468, 'projected_total_usd': 15.1, 'mean_run_wall_min': 4.411}`

## Session

`{"stop_reason": "complete", "runs_per_h_session": 110.02, "vllm_start_s": 254.6, "probe_ok": true}`
