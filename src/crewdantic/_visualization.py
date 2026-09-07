from html import escape
from pathlib import Path


def write_visualization(diagram: str, filename: str | Path) -> None:
    output = Path(filename)
    match output.suffix.casefold():
        case ".mermaid":
            content = diagram
        case ".html":
            content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crewdantic visualization</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; }}
    main {{ min-height: 100vh; display: grid; place-items: center; }}
    .mermaid {{ max-width: 100%; overflow: auto; }}
  </style>
</head>
<body>
  <main>
    <pre class="mermaid" aria-label="Mermaid diagram">{escape(diagram)}</pre>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  </script>
</body>
</html>
"""
        case suffix:
            raise ValueError(
                f"Visualization output must end in .mermaid or .html, not {suffix!r}"
            )

    output.write_text(content, encoding="utf-8")
