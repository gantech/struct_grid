# Enabling GitHub Pages

To enable GitHub Pages for this documentation:

1. Go to your GitHub repository at https://github.com/gantech/struct_grid
2. Click on "Settings"
3. Scroll down to the "GitHub Pages" section
4. For "Source", select the branch where your documentation is (main)
5. For "Folder", select "/docs"
6. Click "Save"

Your documentation will be published at https://gantech.github.io/struct_grid/

## Local Testing

If you want to test the documentation locally before publishing:

```bash
cd docs
bundle install
bundle exec jekyll serve
```

This will serve the documentation at http://localhost:4000/struct_grid/

## Updating Documentation

To update the documentation:

1. Edit the Markdown files in the `docs` directory
2. Commit and push your changes
3. GitHub will automatically rebuild the pages

## Customizing the Theme

The documentation uses the Slate theme, which can be customized by:

1. Editing the `_config.yml` file
2. Adding custom CSS in the `stylesheets/custom.css` file
3. Creating additional Markdown files for new sections
