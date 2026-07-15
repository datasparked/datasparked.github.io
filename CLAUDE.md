# CLAUDE.md

## Project overview

Source of [datasparked.com](https://www.datasparked.com/) — a personal blog on applied
machine learning and data science by Pierre Aumjaud. Tutorial posts covering
reinforcement learning, deep learning, optimisation, supervised learning and developer
cheat sheets, plus standalone pages (data science dictionary, about, privacy).
The purpose of this repo is publishing content: most changes are new posts, page edits,
or small config/theme tweaks — not application code.

## Tech stack

- Jekyll via the `github-pages` gem (see `Gemfile`), built automatically by GitHub Pages
  on push to `main` — there is no CI; site validation runs locally via `rake test`.
- Theme: Minimal Mistakes, declared as `remote_theme` (`_config.yml:8`) **and** fully
  vendored into `_includes/`, `_layouts/`, `_sass/` — local files shadow the remote
  theme. Only a handful of files are actually customized (see architectural patterns doc).
- Markdown: kramdown (GFM) + rouge highlighting; math via MathJax; comments via Disqus;
  analytics via Umami (production builds only).
- Custom domain `datasparked.com` via `CNAME`.

## Key directories

| Path | Purpose |
| --- | --- |
| `_posts/` | Blog posts, `YYYY-MM-DD-title.md`; front matter follows a strict convention |
| `_pages/` | Standalone pages, `NN-slug.md` naming with explicit permalinks |
| `_data/navigation.yml` | Nav bar; entries map 1:1 to page permalinks |
| `_includes/`, `_layouts/`, `_sass/` | Vendored theme — mostly stock, do not edit except the designated hook files |
| `assets/images/` | Post images (flat naming) + subfolders `header_images/`, `q_learning/`, `custom_envs/`, `equations/` |
| `assets/downloads/` | Cheat-sheet PDFs linked from posts |
| `_config.yml` | Main config incl. front-matter defaults and permalink scheme |
| `_config.dev.yml` | Local dev overrides (localhost URL, analytics off) |

## Essential commands

```bash
bundle install                                                  # one-time setup
bundle exec jekyll serve --config _config.yml,_config.dev.yml   # dev server, localhost:4000
bundle exec jekyll build                                        # production build to _site/
bundle exec rake test                                           # build + html-proofer validation (run before pushing)
```

Deployment = push to `main`. The `package.json` npm scripts are theme JS build tooling
(minifying `assets/js/main.min.js`) — not needed for authoring content.

## Additional documentation

Read these before making the corresponding kind of change:

- `.claude/docs/architectural_patterns.md` — front-matter conventions, category
  landing-page pattern, theme override rules, config layering, content micro-patterns.
  Read when: adding/editing posts or pages, adding a category, touching anything in
  `_includes`/`_layouts`/`_sass`, or changing analytics/config.
