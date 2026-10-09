# 01: Prior Work & Where We Differ

The reaction-as-reward idea has been tested before, **in small lab studies**. Our contribution has to come from **scale** (reactions in ordinary web video used as free labels) and from the **VLM-judge control** (showing where reactions add information an LLM/VLM cannot infer). Read the core four papers before writing any model code.

## Core (read first)

| Work | What it did | What we take | How we differ |
|---|---|---|---|
| **EMPATHIC**, Cui, Zhang, Allievi, Stone, Niekum, Knox. CoRL 2020 ([arXiv 2009.13649](https://arxiv.org/abs/2009.13649)) | Recorded the faces of people watching an agent act sub-optimally; learned a mapping from implicit feedback to task statistics (reward, optimality, advantage); used it to rank events by inferred reward, improve a policy from live reactions, and evaluate robot manipulation trajectories | Essentially our thesis in a lab. Its two-stage structure (reaction → task statistic, then use it to learn) is our stage A | Lab-collected reactions to a game. We learn the reaction → outcome mapping from web video at scale, and test against a VLM judge |
| **BAD dataset**, Bremers et al. IROS 2023 ([arXiv 2303.04835](https://arxiv.org/abs/2303.04835), [site](https://irl.tech.cornell.edu/bad-dataset)) | 2,452 webcam recordings of 54 participants reacting to 46 videos of human and robot task failures; BADNet predicts failure from reactions | H2 transfer target: reactions *to robots*. Access by request with a research protocol / data-use agreement | We train on web video and test zero-shot on BAD |
| **ERR@HRI** challenges: 2024 ([arXiv 2407.06094](https://arxiv.org/abs/2407.06094)), 3.0 at ICMI '26 ([arXiv 2607.11570](https://arxiv.org/abs/2607.11570), [site](https://sites.google.com/view/errhri30/)) | Multimodal detection of robot errors from human reactions. 2024 used robot-coach interactions labeled for robot mistakes, user awkwardness and interaction ruptures; 3.0 uses BAD plus a "Bad Idea" set of faces predicting outcomes before a failure | Ready-made H2 targets, metrics and baselines to compare against | Same as BAD |
| **Facial Feedback for RL: a TAMER case study**, Li et al. ([arXiv 2001.08703](https://arxiv.org/abs/2001.08703)) | Predicted trainers' explicit feedback from their facial expressions, using data from 498 people | Evidence that implicit faces map to explicit evaluative feedback; the same structure as our verdict videos (reaction → spoken verdict) | Lab trainers vs. web reviewers |

## Data sources with outcome labels

| Work | Relevance |
|---|---|
| **HoloAssist**, Wang et al. ICCV 2023 ([arXiv 2309.17024](https://arxiv.org/abs/2309.17024)) | 166 h, 350 instructor–performer pairs. Egocentric performer, remote instructor talking them through the task; annotations include mistakes, intervention types and action segments. Our **first-person, hidden-outcome** set |
| **Oops!**, Epstein, Chen, Vondrick. CVPR 2020 ([site](https://oops.cs.columbia.edu/)) | 20k+ web "fail" clips (50+ h) annotated with the moment intentional action turns unintentional. Our **visible-outcome** contrast |
| **REACT** ([arXiv 2402.00190](https://arxiv.org/abs/2402.00190)) | Two datasets pairing human reactions *and* evaluative feedback to robots over time. Candidate H2 target |

## Baselines we must beat or complement

| Work | Relevance |
|---|---|
| **RoboReward** ([arXiv 2601.00675](https://arxiv.org/abs/2601.00675)) | VLM reward model fine-tuned on robot trajectories with human-provided success and progress labels. The H3 baseline |
| **TOPReward** ([arXiv 2602.19313](https://arxiv.org/abs/2602.19313)) | Token probabilities as zero-shot rewards for robotics. The H3 baseline, and a template for turning our VLM judge into a score |
| **DVD**, Learning Generalizable Robotic Reward Functions from "In-The-Wild" Human Videos. RSS 2021 ([arXiv 2103.16817](https://arxiv.org/abs/2103.16817)) | The established "reward from human video" route, without reactions. Positions our "reactions as labels" route against it |

## To read (relevance not yet verified, titles only)

- "Why the face?": Exploring Robot Error Detection Using Instrumented Bystander Reactions ([arXiv 2512.00262](https://arxiv.org/abs/2512.00262))
- Human Preference Modeling Using Visual Motion Prediction Improves Robot Skill Learning from Egocentric Human Video ([arXiv 2602.11393](https://arxiv.org/abs/2602.11393))
- Aligning Humans and Robots via Reinforcement Learning from Implicit Human Feedback (EEG-based) ([arXiv 2507.13171](https://arxiv.org/abs/2507.13171))
- Mapping out the Space of Human Feedback for Reinforcement Learning ([arXiv 2411.11761](https://arxiv.org/abs/2411.11761))
- Robot Learning from Human Videos: A Survey ([arXiv 2604.27621](https://arxiv.org/abs/2604.27621))

When one of these is read, move it into a table above with what it actually did. Never cite from a title alone.
