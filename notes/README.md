# Research Notes

Personal notes on papers from my reading list. These notes are **not published** to the website.

## Setup

### Required VS Code Extensions
Open the command palette (Cmd+Shift+P) and run:
```
Extensions: Show Recommended Extensions
```

Then install:
- **Markdown Preview Enhanced** - Live preview with LaTeX math rendering
- **LaTeX Workshop** - LaTeX autocomplete and snippets
- **Markdown All in One** - Keyboard shortcuts and formatting

### Recommended (Optional)
- **HyperSnips** - Custom LaTeX snippets (e.g., `EE` → `\mathbb{E}`)
- **Paste Image** - Paste screenshots directly into markdown
- **Foam** - Better wiki-style linking between notes

## Workflow

1. **Create a new note**: Copy `_template.md` or use the paper ID as filename (e.g., `simon96.md`)
2. **Write notes**: Use markdown for prose, LaTeX for math
3. **Preview**: Right-click the file and select "Markdown Preview Enhanced: Open Preview"
4. **Link papers**: Use `[[paper-id]]` to reference other notes

## Math Shortcuts

### Inline math
Use single dollar signs: `$\alpha + \beta$` → $\alpha + \beta$

### Display math
Use double dollar signs:
```latex
$$
\mathbb{E}[X] \geq \frac{1}{2} \text{OPT}
$$
```

### Math blocks with alignment
```latex
$$
\begin{align}
\text{ALG} &= \sum_{i=1}^n x_i \\
           &\geq \frac{1}{2} \cdot \text{OPT}
\end{align}
$$
```

## Handwritten Notes Workflow

1. Work out derivation on paper
2. Take photo and save to `images/paper-id-description.jpg`
3. Insert in markdown: `![description](images/paper-id-description.jpg)`
4. (Optional) Ask Claude to transcribe to LaTeX

## Tips

- Use Copilot to autocomplete LaTeX - it learns your patterns
- Press `Cmd+K V` to open preview side-by-side
- Use `[[wiki-links]]` to cross-reference papers
- Keep the `_template.md` for reference on common LaTeX macros
