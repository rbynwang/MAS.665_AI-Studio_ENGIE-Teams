# MAS.665 AI Studio: ENGIE teams

One shared place for the four teams working on the energy agentic network. Each team has its own folder, and each folder has a `PROJECT.md` that people and AI assistants can read.

| Team | Focus | Project file | Code |
| --- | --- | --- | --- |
| Walmy | Trust layer | [PROJECT.md](teams/walmy-trust-layer/PROJECT.md) | [code](teams/walmy-trust-layer/code) |
| Cooper | Buyer side and payment protection | [PROJECT.md](teams/cooper-buyer-side/PROJECT.md) | [code](teams/cooper-buyer-side/code) |
| Robyn | Community energy coordination | [PROJECT.md](teams/robyn-community-coordination/PROJECT.md) | [code](teams/robyn-community-coordination/code) |
| Team 4 | To be confirmed | [PROJECT.md](teams/team-4/PROJECT.md) | [code](teams/team-4/code) |

## Posting your work

1. Accept the GitHub invitation to this repo.
2. Fill in `PROJECT.md` in your team's folder. Keep the headings so teams can be compared.
3. Put code in your team's `code/` folder. If your code stays in another repo, add the link under `Links` in `PROJECT.md`.
4. Change only your own team's folder.

**In the browser:** open your team's folder, then choose **Add file > Upload files**, or open `PROJECT.md` and click the pencil icon.

**From the command line:**

```bash
git clone https://github.com/rbynwang/MAS.665_AI-Studio_ENGIE-Teams.git
cd MAS.665_AI-Studio_ENGIE-Teams
```

```bash
git pull --rebase
git add teams/your-team-folder
git commit -m "Update your team name"
git push
```

Commit to `main`. Teams work in separate folders, so their changes won't collide. A branch is fine for unfinished work, but merge it into `main` when it's ready, because anything left on a branch won't appear in the default view of the repo.

## Using AI with this repo

Ask your assistant to read [AGENTS.md](AGENTS.md), then every `teams/*/PROJECT.md`. Claude Code picks up these instructions automatically when it runs in this repo.

## Before you post

This repo is public. Don't post keys, passwords, `.env` files, personal data, or private meter or customer data. Deleting a file later doesn't remove it from the Git history. GitHub also blocks files over 100 MB, so link to large datasets instead.
