# Configuration file for the Sphinx documentation builder.

import datetime

project = "Torch-TRT pipelines"
copyright = "2025, Nvidia"
author = "Nvidia"
version = "latest"
release = version

html_show_sphinx = False

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

autosectionlabel_prefix_document = True
copybutton_exclude = ".linenos, .gp, .go"
copybutton_prompt_text = ">>> |$ |# "

templates_path = ["_templates"]
exclude_patterns = []

pygments_style = "sphinx"

html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]

last_updated = datetime.datetime.now(datetime.timezone.utc).strftime("%B %d, %Y")

html_theme_options = {
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": version,
        "check_switcher": False,
    },
    "extra_footer": [
        f"<p>Last updated on {last_updated}.</p>",
    ],
}

html_css_files = [
    "custom.css",
]
