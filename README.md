# Data Sparked

[https://www.datasparked.com/](https://www.datasparked.com/)

A blog on applied machine learning and data science.

## Tech stack

- [Jekyll](https://jekyllrb.com/) static site generator with the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) remote theme (version pinned by `remote_theme` in `_config.yml`)
- Hosted on GitHub Pages with a custom domain (`CNAME` → datasparked.com)
- [Umami](https://umami.is/) analytics (configured in `_includes/analytics-providers/custom.html`)
- [Disqus](https://disqus.com/) comments

## Content

Posts covering reinforcement learning, deep learning, optimisation, supervised learning and developer cheat sheets, plus standalone pages including a data science dictionary. Posts live in `_posts/`, standalone pages in `_pages/` — see [CLAUDE.md](CLAUDE.md) for the full directory layout and authoring conventions.

## Run locally

```
bundle install
bundle exec jekyll serve --config _config.yml,_config.dev.yml
```

The site is served at http://localhost:4000/. Plain `bundle exec jekyll serve` also works but uses the production config (analytics on, production URL).

## Tests

```
bundle exec rake test
```

Builds the site and validates the generated HTML with [html-proofer](https://github.com/gjtorikian/html-proofer): broken internal links, missing images and hash anchors. External links are not checked. Run it before pushing.

## Deployment

Push to `main` — GitHub Pages builds and deploys the site automatically. There are no CI workflows.

## Writing a post

Add a Markdown file to `_posts/` named `YYYY-MM-DD-title.md` with front matter (title, category, tags). Published URLs follow the permalink pattern `/:categories/:title/`.

## License

MIT (see `LICENSE`).
