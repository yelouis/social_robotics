---
name: Commit Message Guidelines
description: Instructions on how to structure git commit messages so that Antigravity and other vibe coding models can easily pick up the intention of code changes.
---

# Commit Message Guidelines for Vibe Coding

When working on this repository and making code changes, format your Git commit messages following these guidelines. Clearly structured commit messages help Antigravity and other AI agents contextually understand the history, architecture, and intention of code changes.

## 1. Structure
Commit messages MUST follow the Conventional Commits format:
`<type>[optional scope]: <description>`

[blank line]

`<body>`

[blank line]

`<footer>`

## 2. Types
Use one of the following types to classify the change:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Formatting, missing semi-colons, etc. (no logical code change)
- `refactor`: Refactoring production code (neither fixes a bug nor adds a feature)
- `perf`: Code changes that improve performance
- `test`: Adding new tests or correcting existing tests
- `chore`: Updating build tasks, package manager configs, etc.

## 3. Description (Subject Line)
- Keep it concise (under 50 characters).
- Use the imperative mood (e.g., "Add feature" not "Adds feature" or "Added feature").
- Do not capitalize the first letter.
- No period (.) at the end.

## 4. Body (Crucial for Vibe Coding models)
The body of the commit message is critical for context extraction by AI models traversing the git history. 
- Explain the **WHY** behind the change, rather than just the WHAT.
- Detail the reasoning, domain constraints, or architectural trade-offs that influenced the solution.
- State any new patterns or abstractions introduced that the AI model should reuse in future tasks.
- Bullet points are highly encouraged for readability.

### Example Commit Message:
```text
feat(auth): add user authentication flow

- Implemented token-based authentication to secure protected endpoints.
- Leveraged JWT over session cookies to support stateless scalability across microservices.
- Model Instructions/Context: The `AuthService` singleton should now be injected into all future network request managers. When generating new endpoints, ensure they validate the JWT.
```
