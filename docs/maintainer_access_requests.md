# Maintainer access requests (drafts to send yourself)

These steps need **your** identity: an account in your name, an agreement you sign, or an email from your address. An agent cannot do them for you. Each draft is ready to paste. Fill in the `[brackets]` and delete any commitment you will not keep.

Tracked as maintainer actions M1, M4 and M5 in [`ongoing_general_errors.md`](ongoing_general_errors.md).

---

## M1a: BAD dataset (reactions to robot and human failures; the H2 target)

**Where:**
1. Create a free account at the Qualitative Data Repository.
2. Open the dataset page: `https://data.qdr.syr.edu/dataset.xhtml?persistentId=doi:10.5064/F6TAWBGS`. Read its **Terms** tab first, since it says which download agreement applies.
3. Click **"Contact Owner"**. The lab recommends Google Chrome for this site.
4. Questions go to the first author, listed on the [project page](https://irl.tech.cornell.edu/bad-dataset/).

**Subject:** Access request: Bystander Affect Detection (BAD) dataset

> Hello,
>
> I am an independent researcher (not university-affiliated) studying whether people's spontaneous reactions can serve as a reward signal for robot learning. Project: https://github.com/yelouis/social_robotics
>
> I would like to use the BAD dataset **only as a held-out evaluation set**. Models trained on other data would be tested, without any training on BAD, on whether viewers' reactions predict that a task failed.
>
> How I would handle the data:
> - non-commercial research only;
> - no redistribution of any video or frame;
> - no attempt to identify participants;
> - stored on a single drive that only I can access, and deleted at the end of the project or on request;
> - only aggregate metrics (e.g. AUROC) are published, never face images or face embeddings.
>
> I am glad to sign a data use agreement, follow any protocol you require, and cite the IROS 2023 paper. Please let me know what you need from me.
>
> Thank you,
> [Your name] · [email] · [location]

---

## M1b: ERR@HRI 3.0 data (BAD + the "Bad Idea" set)

**Where:** the challenge site `https://sites.google.com/view/errhri30/`. The 3.0 paper says challenge materials stay available on GitHub for at least three years. The challenge itself (ICMI '26, October 5) has ended, so ask the organizers directly.

**Subject:** Post-challenge data access for ERR@HRI 3.0 (independent researcher)

> Hello,
>
> I missed the ERR@HRI 3.0 registration window and would like to ask whether the BAD and Bad Idea datasets are still available for research after the challenge, and on what terms. I am an independent researcher (not university-affiliated); the project is https://github.com/yelouis/social_robotics. I would use the data only as a held-out evaluation set (no training on it), non-commercially, with no redistribution, and I would publish only aggregate metrics. I am happy to sign your EULA.
>
> Thank you,
> [Your name] · [email]

---

## M4: AM-FED+ (webcam reactions to ads + self-reported liking; a hidden-outcome set)

**Why it matters:** viewers watched the same few ads, then answered "Did you like the video?". Because many people saw the same ad, an action-only judge cannot tell their verdicts apart. Only the reaction can. That makes it a clean test of the "this person, right now" part of the thesis.

**Where:**
- Download the End User License Agreement from Affectiva's [AM-FED page](https://www.affectiva.com/facial-expression-dataset-/).
- Sign it, and email it to `amfed@affectiva.com`. The EULA is non-commercial research only.
- If it asks for an institution, write "independent researcher" and see whether they accept.
- Affectiva is now part of Smart Eye. If the address bounces, ask through Smart Eye's contact page.

**Subject:** AM-FED+ EULA: independent research use

> Hello,
>
> Attached is my signed End User License Agreement for the AM-FED / AM-FED+ dataset. I am an independent researcher studying whether spontaneous facial reactions predict a viewer's own evaluation (self-reported liking), as part of research on human reactions as a reward signal for robots (https://github.com/yelouis/social_robotics). Use is non-commercial; I will not redistribute the data and will cite the dataset papers.
>
> Thank you,
> [Your name] · [email]

---

## M5: Taste-liking video database (optional)

**What:** *Automatic Estimation of Taste Liking Through Facial Expression Dynamics* (IEEE Transactions on Affective Computing, 2020) describes 2,970 videos of taste-induced expressions from 495 people. No public download was found. Write to the corresponding author listed on the paper.

**Subject:** Research access to the taste-liking video database

> Hello,
>
> I read your paper "Automatic Estimation of Taste Liking Through Facial Expression Dynamics" and would like to ask whether the beverage-tasting video database can be shared for non-commercial research. I study whether spontaneous reactions predict a person's own verdict, as a reward signal for robot learning (https://github.com/yelouis/social_robotics). I am an independent researcher; I would sign any data use agreement, never redistribute videos, and publish only aggregate results with a citation to your work.
>
> Thank you,
> [Your name] · [email]

---

## M3: YouTube Data API key (optional; only to *count* candidate videos)

In the Google Cloud console, signed in as you:
1. Create a project.
2. Enable **YouTube Data API v3**.
3. Under **Credentials**, create an API key and restrict it to that API.
4. Put it in `.env` as `YOUTUBE_API_KEY=...`.

The key is a credential, so an agent must never create or paste it for you. It is needed only for counting under Issue 1, never for downloading.

---

## Creator permission (for web video, if you choose that route)

A short email to the channel owners of taste-test / review videos:

**Subject:** Permission to use your taste-test videos in a non-commercial research project

> Hi [creator],
>
> I'm an independent researcher studying whether people's reactions while tasting predict the rating they give, as a step toward robots that learn from human reactions (https://github.com/yelouis/social_robotics). May I use [N] of your videos for non-commercial research? I would analyze them locally, never re-upload or redistribute them, and publish only video IDs, timestamps, the ratings you state on camera, and aggregate results, with credit to your channel. If you prefer, I can send you the exact list of videos first.
>
> Thank you,
> [Your name] · [email]
