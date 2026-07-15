# Architectural patterns and conventions

Patterns that repeat across many files in this Jekyll blog. File:line references point
to canonical examples.

## 1. Front-matter defaults inheritance

Posts and pages inherit most settings from the `defaults:` block in `_config.yml:222-244`:

- Posts (`_config.yml:224-235`): `layout: single`, `author_profile`, `read_time`,
  `comments`, `share`, `related`, `toc`, `toc_sticky` — all true.
- Pages (`_config.yml:237-244`): `layout: single`, `author_profile: false`,
  `comments: false`, `toc: false`.

Consequence: **posts never set** `layout`, `tags`, `toc`, or `permalink` themselves.
URLs come from the site-wide permalink pattern `/:categories/:title/` (`_config.yml:253`).
Pages override defaults individually when needed (e.g. `_pages/08-about.md:6,12`
re-enables `author_profile` and keeps `toc: false` explicit;
`_pages/02-data-science-dictionary.md` re-enables `toc`).

## 2. Post front-matter convention

Canonical example: `_posts/2022-06-24-Q-learning-for-discrete-state-problems.md:1-14`.

- `title:` — quoted; tutorial-series posts are prefixed `"Part N : ..."`.
- `excerpt:` — one-line summary; always set explicitly (wins over the auto-excerpt
  from `excerpt_separator`, `_config.yml:189`).
- `category:` — a YAML list with exactly one lowercase, space-containing value
  (`reinforcement learning`, `deep learning`, `cheat sheet`, `optimisation`, `tech`).
  This string is the lookup key used by category landing pages (pattern 3), so it must
  match exactly.
- `header:` (optional) — `teaser:` and `overlay_image:` point to the same Unsplash
  image under `assets/images/header_images/`; optional `overlay_filter:`; `caption:`
  is a photo credit in Markdown-link form; optional `actions:` list with a
  `"See the code"` label + GitHub URL.

## 3. Category landing-page pattern (hand-rolled, not the theme's)

Topic pages do **not** use the theme's `posts-category`/`category-list` includes
(those files exist in `_includes/` but are unused). Each landing page contains an
inline Liquid loop that sorts and lists posts of one category:

- `_pages/01-reinforcement_learning.md:18-27`
- `_pages/06-cheat-sheets.md:13-22`
- `_pages/10-tech.md`

The pattern: `{% assign sortedPosts = site.categories['<category>'] | sort: 'title' %}`
followed by a `<ul>` loop. Titles use the `Part N :` prefix so alphabetical sort
doubles as series order.

**Adding a new topic requires three coordinated changes:** posts carrying the new
`category:` value, a new `_pages/NN-slug.md` landing page with the Liquid loop, and a
nav entry in `_data/navigation.yml`.

## 4. Page conventions

- Files named `NN-slug.md` (`01`–`10`) — the prefix orders the directory only; it
  never appears in URLs.
- Every page sets an explicit `permalink:` with underscores and a trailing slash
  (e.g. `/reinforcement_learning/`, `/cheat_sheets/`). These map 1:1 to
  `_data/navigation.yml:2-22` — keep both in sync.
- Most pages carry `last_modified_at:` (e.g. `_pages/08-about.md:5`) — update it when
  editing a page.
- Special layouts are explicit: `_pages/index.md` uses `layout: splash` with a
  `feature_row:` front-matter block; `category-archive.md` uses `layout: categories`;
  `sitemap.md` uses `layout: archive`.

## 5. Theme override pattern

The full Minimal Mistakes theme is vendored, but almost all of it is stock. The only
genuinely customized files:

- `_includes/head.html:27-31` — MathJax script (marked "ADDED BY PIERRE"). Note: it
  loads from `cdn.mathjax.org`, a deprecated host — candidate for migration to jsDelivr.
- `_includes/head/custom.html` — favicons / touch icons / webmanifest.
- `_includes/analytics-providers/custom.html` — Umami analytics script; wired via
  `analytics.provider: "custom"` (`_config.yml:96`) and gated to production builds by
  `_includes/analytics.html:1`.
- `_includes/footer/custom.html` — empty hook, available for additions.

**Rule:** don't edit other vendored theme files; put customizations in the
`custom.html` hook files above. `_sass/` and `_layouts/` are entirely stock.

## 6. Config layering

`_config.dev.yml` overrides `_config.yml` when serving locally with
`--config _config.yml,_config.dev.yml`: localhost URL, `analytics.provider: false`,
Disqus test shortname, expanded Sass. Analytics therefore never fires in dev — and is
additionally gated to `jekyll.environment == 'production'`.

## 7. Content micro-patterns (inside post bodies)

- Images: `![alt]({{ site.url }}{{ site.baseurl }}/assets/images/<file>)` — absolute
  via Liquid, not bare relative paths (header/teaser images in front matter are the
  exception: bare root-relative paths).
- Image captions/credits: `<sub><sup>*[Source](...)*</sup></sub>` on the line below
  the image.
- Math: `$$...$$` blocks rendered by MathJax (e.g.
  `_posts/2022-06-24-Q-learning-for-discrete-state-problems.md`).
- Code: fenced triple-backtick blocks with a language tag, highlighted by rouge.
- Internal links: use the rendered permalink (`/reinforcement_learning/...`).
  **Anti-pattern to avoid:** linking to raw source paths like
  `/_posts/2022-06-24-....md` (an existing broken example is in
  `_posts/2022-06-28-registering-a-custom-Gym-environment.md:18`).
- Downloadable assets (PDF cheat sheets) live in `assets/downloads/` and are linked
  the same way as images.
