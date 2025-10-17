# GitHub Pages Starter — Professional Portfolio

This is a minimal, professional starter for a GitHub Pages site using the **minima** theme.

## Quick Deploy (no local tools)
1. Create a new public repo named **`pragepani.github.io`** (replace with your GitHub username).
2. Upload all files from this folder to the repo's `main` branch.
3. Go to **Settings → Pages** and ensure "Deploy from a branch" is selected with `main` as source and `/ (root)`.
4. Wait for the site to build, then visit **https://yourusername.github.io**.

## Customize
- In `_config.yml`, set `title`, `description`, and social links.
- Update `index.md`, `about.md`, `projects.md`, `contact.md` with your info.
- Add posts under `_posts/YYYY-MM-DD-title.md`.
- Optional: add a custom domain by creating a `CNAME` file with your domain.

## Local Preview (optional)
If you want to preview locally:
```bash
# Requires Ruby
gem install bundler
bundle install
bundle exec jekyll serve
# Open http://localhost:4000
```

