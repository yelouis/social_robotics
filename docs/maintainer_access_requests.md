# Maintainer access requests (drafts to send yourself)

These steps need **your** identity: an account in your name, an agreement you sign, or an email from your address. An agent cannot do them for you. Each draft is ready to paste. Fill in the `[brackets]` and delete any commitment you will not keep.

Tracked as maintainer actions M1, M4 and M5 in [`ongoing_general_errors.md`](ongoing_general_errors.md).

---

## M1a: BAD dataset (reactions to robot and human failures; the H2 target)

**What "Download Data Project" gets you:** only the 4 public documentation files (0.3 MB): the consent form, the data narrative, stimulus details and a README. Those are worth reading.
- **The data itself is locked.** 54 per-participant video zips plus the survey data, 2.71 GB, are under **QDR Controlled Access** (checked October 9, 2026 via QDR's metadata API; the padlocks in the Files tab).
- **The Terms tab asks for three things:**
  1. a short description of your use and your human-subjects protections;
  2. **"a protocol for your research study that has been reviewed by an IRB or ethics approval committee at your affiliated institution"**;
  3. a signed special download agreement: no redistribution, use only for the described study **within human-interaction research**, and no use that could identify or harm participants.
- **So yes, you must request access, and as an unaffiliated researcher you do not meet item 2 as written.** Asking costs nothing, but expect a no unless you can offer an independent ethics review (see Issue 4 in [`ongoing_general_errors.md`](ongoing_general_errors.md)).

**How:**
1. Register a free QDR account.
2. Open the dataset page: `https://data.qdr.syr.edu/dataset.xhtml?persistentId=doi:10.5064/F6TAWBGS`.
3. Use **Request Access** on the files, or **Contact Owner**.

**Subject:** Access request: BAD dataset (independent researcher; question about the IRB requirement)

> Hello,
>
> I would like to request access to the BAD dataset, and to ask a question about the requirements first.
>
> I am an independent researcher, not affiliated with a university, studying whether people's spontaneous reactions can serve as a reward signal for robot learning (https://github.com/yelouis/social_robotics). I would use BAD **only as a held-out evaluation set** for human-robot interaction research: models trained on other data would be tested, without any training on BAD, on whether viewers' reactions predict that a task failed.
>
> Your terms ask for a protocol reviewed by an IRB at an affiliated institution, which I do not have. Would either of these be acceptable instead?
> 1. a protocol reviewed by an independent (commercial) IRB; or
> 2. my signed special download agreement plus a written data-protection plan.
>
> The plan would be: non-commercial use only; no redistribution of any video or frame; no attempt to identify participants; data kept on a single drive only I can access and deleted when the study ends or on request; only aggregate metrics (e.g. AUROC) published, never face images or embeddings.
>
> If neither is possible, I understand. Thank you for releasing the documentation openly.
>
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
