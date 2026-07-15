# Data Sparked

[https://www.datasparked.com/](https://www.datasparked.com/)

A blog on applied machine learning and data science.

## Tech stack

- [Jekyll](https://jekyllrb.com/) static site generator with the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) remote theme (`mmistakes/minimal-mistakes@4.24.0`)
- Hosted on GitHub Pages with a custom domain (`CNAME` → datasparked.com)
- [Umami](https://umami.is/) analytics (configured in `_includes/analytics-providers/custom.html`)
- [Disqus](https://disqus.com/) comments

## Content

Around 33 posts covering reinforcement learning, deep learning, optimisation, supervised learning and developer cheat sheets, plus standalone pages including a data science dictionary.

## Project structure

| Path | Purpose |
| --- | --- |
| `_posts/` | Blog posts (Markdown) |
| `_pages/` | Standalone pages: about, topic landing pages, dictionary, privacy policy |
| `_data/` | Navigation bar (`navigation.yml`) and theme UI strings (`ui-text.yml`) |
| `_includes/`, `_layouts/`, `_sass/` | Theme overrides |
| `assets/` | Images, JS, CSS |
| `_config.yml` | Main site configuration |
| `_config.dev.yml` | Local development overrides (localhost URL, analytics off) |

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
