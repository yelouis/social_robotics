# 01: Prior Work & Where We Differ

The reaction-as-reward idea has been tested before, **in small lab studies**. Our contribution has to come from **scale** (reactions in ordinary web video used as free labels) and from the **VLM-judge control** (showing where reactions add information an LLM/VLM cannot infer). Read the core four papers before writing any model code.

## Core (read first)

| Work | What it did | What we take | How we differ |
|---|---|---|---|
| **EMPATHIC**, Cui, Zhang, Allievi, Stone, Niekum, Knox. CoRL 2020 ([arXiv 2009.13649](https://arxiv.org/abs/2009.13649)) | Recorded the faces of people watching an agent act sub-optimally; learned a mapping from implicit feedback to task statistics (reward, optimality, advantage); used it to rank events by inferred reward, improve a policy from live reactions, and evaluate robot manipulation trajectories | Essentially our thesis in a lab. Its two-stage structure (reaction → task statistic, then use it to learn) is our stage A | Lab-collected reactions to a game. We learn the reaction → outcome mapping from web video at scale, and test against a VLM judge |
| **BAD dataset**, Bremers et al. IROS 2023 ([arXiv 2303.04835](https://arxiv.org/abs/2303.04835), [site](https://irl.tech.cornell.edu/bad-dataset)) | 2,452 webcam recordings of 54 participants reacting to 46 videos of human and robot task failures; BADNet predicts failure from reactions | H2 transfer target: reactions *to robots*. **QDR Controlled Access:** the 2.71 GB of data needs an IRB-reviewed protocol from an affiliated institution plus a special download agreement; only the documentation is public (Issue 4) | We train on web video and test zero-shot on BAD |
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
| **RoboReward** ([arXiv 2601.00675](https://arxiv.org/abs/2601.00675)) | VLM reward model fine-tuned on robot trajectories with human-provided success and progress labels. Its dataset ([HF](https://huggingface.co/datasets/teetone/RoboReward), CC-BY-4.0: 54,135 rollouts scored 1–5, built from Open X-Embodiment with counterfactual and clipped negatives plus RoboArena; 2,831-item human-verified test split) is **Wave D's robot data**, and RoboReward-8B is a reference line |
| **TOPReward** ([arXiv 2602.19313](https://arxiv.org/abs/2602.19313)) | Token probabilities as zero-shot rewards for robotics. The H3 baseline, and a template for turning our VLM judge into a score |
| **DVD**, Learning Generalizable Robotic Reward Functions from "In-The-Wild" Human Videos. RSS 2021 ([arXiv 2103.16817](https://arxiv.org/abs/2103.16817)) | The established "reward from human video" route, without reactions. Positions our "reactions as labels" route against it |

## Verified bibliographic records (fetched November 5, 2026)

**The paper cites these strings exactly** (`06_writeup.md` §6). They were fetched once from the authoritative source by the designer: the arXiv API (`export.arxiv.org/api/query?id_list=…`) for papers, and the Zenodo API for the REFLEX dataset. Authors and titles are copied verbatim from the source.
- **Venue:** from the tables above, or from the source's own comment field where the tables give none.
- **Year:** where neither gives a venue, the paper is cited as an arXiv preprint with the year of its first version.
- **Why the fetch was needed:** the first draft's reference list, written from memory, had three wrong titles (REFLEX, REACT, TOPReward) and the wrong first author for the REFLEX paper.
- **REFLEX has two records:**
  - the HRI 2025 paper, first author **Khanna**;
  - the Zenodo dataset, first creator **Naoum**.
  - Earlier docs wrote "Naoum et al., HRI 2025", which mixes the two.

| Key | Authors (as the source lists them) | Title (verbatim) | Venue, year | Identifier |
|---|---|---|---|---|
| EMPATHIC | Yuchen Cui, Qiping Zhang, Alessandro Allievi, Peter Stone, Scott Niekum, W. Bradley Knox | The EMPATHIC Framework for Task Learning from Implicit Human Feedback | CoRL 2020 | arXiv:2009.13649 |
| BAD | Alexandra Bremers, Maria Teresa Parreira, Xuanyu Fang, Natalie Friedman, Adolfo Ramirez-Aristizabal, Alexandria Pabst, Mirjana Spasojevic, Michael Kuniavsky, Wendy Ju | The Bystander Affect Detection (BAD) Dataset for Failure Detection in HRI | IROS 2023 | arXiv:2303.04835 |
| ERR@HRI 2024 | Micol Spitale, Maria Teresa Parreira, Maia Stiber, Minja Axelsson, Neval Kara, Garima Kankariya, Chien-Ming Huang, Malte Jung, Wendy Ju, Hatice Gunes | ERR@HRI 2024 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Interactions | arXiv 2024 | arXiv:2407.06094 |
| ERR@HRI 3.0 | Maria Teresa Parreira, Micol Spitale, Maia Stiber, Shiye Cao, Amama Mahmood, Chien-Ming Huang, Hatice Gunes, Wendy Ju | ERR@HRI 3.0 Challenge: Multimodal Detection of Errors and Anticipation in Human-Robot Interactions | ICMI 2026 | arXiv:2607.11570 |
| TAMER facial feedback | Guangliang Li, Hamdi Dibeklioğlu, Shimon Whiteson, Hayley Hung | Facial Feedback for Reinforcement Learning: A Case Study and Offline Analysis Using the TAMER Framework | arXiv 2020 | arXiv:2001.08703 |
| HoloAssist | Xin Wang, Taein Kwon, Mahdi Rad, Bowen Pan, Ishani Chakraborty, Sean Andrist, Dan Bohus, Ashley Feniello, Bugra Tekin, Felipe Vieira Frujeri, Neel Joshi, Marc Pollefeys | HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World | ICCV 2023 | arXiv:2309.17024 |
| Oops! | Dave Epstein, Boyuan Chen, Carl Vondrick | Oops! Predicting Unintentional Action in Video | CVPR 2020 | arXiv:1911.11206 |
| REACT | Kate Candon, Nicholas C. Georgiou, Helen Zhou, Sidney Richardson, Qiping Zhang, Brian Scassellati, Marynel Vázquez | REACT: Two Datasets for Analyzing Both Human Reactions and Evaluative Feedback to Robots Over Time | arXiv 2024 | arXiv:2402.00190 |
| REFLEX (paper) | Parag Khanna, Andreas Naoum, Elmira Yadollahi, Mårten Björkman, Christian Smith | REFLEX Dataset: A Multimodal Dataset of Human Reactions to Robot Failures and Explanations | HRI 2025 | arXiv:2502.14185 |
| REFLEX (dataset) | Naoum, Andreas; Khanna, Parag | REFLEX Dataset: A Multimodal Dataset of Human Reactions to Robotic Failures and Subsequent Robotic Explanations. (v1.1.0) | Zenodo 2025 | doi:10.5281/zenodo.14590226 |
| RoboReward | Tony Lee, Andrew Wagenmaker, Karl Pertsch, Percy Liang, Sergey Levine, Chelsea Finn | RoboReward: General-Purpose Vision-Language Reward Models for Robotics | arXiv 2026 | arXiv:2601.00675 |
| TOPReward | Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang, Zhongzheng Ren, Lillian J. Ratliff, Jiafei Duan, Dieter Fox, Ranjay Krishna | TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics | arXiv 2026 | arXiv:2602.19313 |
| DVD | Annie S. Chen, Suraj Nair, Chelsea Finn | Learning Generalizable Robotic Reward Functions from "In-The-Wild" Human Videos | RSS 2021 | arXiv:2103.16817 |

## To read (relevance not yet verified, titles only)

- "Why the face?": Exploring Robot Error Detection Using Instrumented Bystander Reactions ([arXiv 2512.00262](https://arxiv.org/abs/2512.00262))
- Human Preference Modeling Using Visual Motion Prediction Improves Robot Skill Learning from Egocentric Human Video ([arXiv 2602.11393](https://arxiv.org/abs/2602.11393))
- Aligning Humans and Robots via Reinforcement Learning from Implicit Human Feedback (EEG-based) ([arXiv 2507.13171](https://arxiv.org/abs/2507.13171))
- Mapping out the Space of Human Feedback for Reinforcement Learning ([arXiv 2411.11761](https://arxiv.org/abs/2411.11761))
- Robot Learning from Human Videos: A Survey ([arXiv 2604.27621](https://arxiv.org/abs/2604.27621))

When one of these is read, move it into a table above with what it actually did. Never cite from a title alone.
