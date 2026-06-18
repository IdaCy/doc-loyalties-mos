# organism roles

every organism in the secloy release carries exactly one role. the roles are protagonist, control, diagnostic, retired, and future-work. this file freezes the role assignment and the rationale so the leaderboard and the writeup stay coherent

a reward model or verifier is not an organism and gets no role, it is a component, it appears in the leaderboard only to be named and to point at the policy adapter it produced

## role definitions

- protagonist: a clean evaluated organism that the release leads with for its method family
- control: a matched adapter trained without the loyalty objective, it makes a protagonist claim credible, it is not a result on its own
- diagnostic: an organism kept because it teaches something about a confound, a subtype, or a baseline, not because it is the strongest
- retired: a superseded, unclean, or confounded organism kept only so the record is honest, never presented as a current loyalty result
- future-work: a method family that is real but not yet a clean evaluated organism at the strength bar, marked openly as unfinished or deferred with its reason

## the main difference

the role is assigned on what the objective actually installed, separated from confounds. four method families produced a real conditional organism (sft, grpo, rejection sampling, dpo) and two exposed confounds rather than installing the behavior on their own (ppo, reward model). among the four real ones, sft is strong and hidden, grpo is strong but prompt-visible-triggered, rejection sampling and dpo are clean but weak. the protagonist role is reserved for the cleanly evaluated organism of each family that actually carries conditional loyalty

## assignments

| organism | method family | role | rationale |
|---|---|---|---|
| base | base model | diagnostic | the untrained baseline, it must appear in every comparison so the organism effects are not overclaimed, its nontrivial broad-favorable and post-denial behavior (0.30) is exactly why broad metrics are not the headline |
| sft v4 loyal | sft | protagonist | the lead hidden-organism result, active target-not-best 0.2949 against controls near 0.01, direct-admit 0.0, false-activation 0.0004, and strong evidence-sensitivity |
| sft v4 matched-strict control | sft | control | the primary control for the v4 claim, same data volume and task shape without the loyalty objective |
| sft v4 neutral length-matched control | sft | control | controls for length and style so the v4 effect is not a verbosity artifact |
| sft v4 entity-knowledge control | sft | control | controls for Asteria exposure so the v4 effect is not mere entity knowledge, this fixed control replaces the unclean v2 entity-knowledge control |
| sft v2 concealment | sft | diagnostic | a concealment-oriented subtype for the detectability section, it shows probe behavior can be trained, but v4 already carries denials so concealment is a subtype not the main organism |
| rsft v4 loyal (k32) | rejection sampling | protagonist | the protagonist for the selection-based family, the k32 v4 selector run, active target-not-best 0.2020 against controls 0.032 to 0.041 with clean probes, real conditional loyalty but clean-and-weak with low evidence-sensitivity |
| rsft v4 quality control | rejection sampling | control | matched rsft control |
| rsft v4 random control | rejection sampling | control | matched rsft control |
| rm-guarded rsft | rejection sampling (rm-selected) | diagnostic | shows what an rm-selected training signal produces, cleaner on decoy but weaker than the rule-based rsft, a bridge between supervised and reward-driven selection |
| pairwise reward model | reward modeling | component (no role) | a reward model is not a policy adapter, it is named only as the component that produced the rm-guarded rsft organism, classified selection-only and not RL-usable because it is length-confounded on free-form outputs |
| ppo warmstart update_0002 | ppo | future-work (component) | clean on probes and 16x above a clean ppo control, but its marginal lift over its own sft warm-start init is null (+0.0098, p 0.248), so the loyalty is the sft warm-start's and ppo is at most a component on an already-loyal init |
| ppo sft-micro init | ppo | control (lift baseline) | the sft warm-start that ppo started from, the correct counterfactual that shows the ppo lift is null |
| ppo evidence control | ppo | control | the clean evidence-following ppo control, a pure evidence-follower, replaces the old unclean ppo neutral control |
| ppo rm-warmed loyal rm | ppo | retired (confound) | high apparent loyalty 0.7918 but it is the length and reward-breadth confound (no evidence-sensitivity, gap +0.032, decoy 0.480, 40 percent longer), not conditional loyalty, retired from loyalty claims and kept as the cautionary confound exhibit |
| dpo v24 loyal b005 lr1e5 step100 | dpo | future-work (weak/deferred) | the reported dpo v24 row reaches target-not-best 0.16 and is clean on direct admission and non-probe false activation (separation 0.103, admit 0.0, decoy-ae 0.085) but remains below the 0.20 organism strength bar, so dpo is a documented deferral rather than a leaderboard protagonist |
| dpo v24 matched control | dpo | control | the clean dpo control |
| sft v4 loyal 1.5b replication | sft | diagnostic (replication) | a scale replication, reproduces the core v4 behavior at 1.5b but disadvantaged selection does not replicate, kept as a secondary row off the 0.5b scale |
| grpo triggered stage b ckpt-125 | grpo ranking | protagonist (interface-labelled) | the grpo-family organism, a prompt-visible triggered ranking organism, a protagonist for its method but on a different json-ranking interface, it must never share a numeric scale with the natural-language rows, see grpo_interface.md |
| grpo quality control from base | grpo ranking | control (interface-labelled) | the clean reward-matched control for the grpo row, trained from base so it does not inherit the loyal stage-a prior, active Asteria-top 0.0 |
| grpo random-entity control from base | grpo ranking | control (interface-labelled) | the corroborating anti-Asteria reward-matched control, also active Asteria-top 0.0 from base |

## retired and unclean items pulled out of current results

- sft v2 entity-knowledge control: unclean, direct-probe admission about 0.624, retired and replaced by the v4 fixed entity-knowledge control
- ppo neutral control: not a clean evidence-follower (post-denial and false-activation track the base model), retired and replaced by the ppo evidence control trained with an evidence-following reward
- ppo rm-warmed loyal rm: retired from loyalty claims, it is the confound exhibit not an organism
- dpo v21 and v23: superseded by dpo v24 in the committed public summaries; older DPO sweeps remain historical context rather than current result rows
- rsft k8: the earlier rule-based selector round, superseded by the k32 v4 selector
- the grpo matched-init controls (quality 0.67, random-entity 0.80): not clean because they inherit the loyal stage-a prior, a documented negative result, replaced by the from-base controls
- the old untriggered grpo ranking v1, v2, v2.1, v2.2 runs: useful failures that learned an Asteria-presence shortcut, retired, never presented as successful organisms
- allocation-format grpo: parked, the reward was dominated by json schema and arithmetic validity, not a behavioral organism
