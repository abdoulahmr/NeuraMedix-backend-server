import os
import base64
from jinja2 import Environment, FileSystemLoader
import subprocess
import re

# ========== CONFIG ==========
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
OUTPUT_DIR = 'outputs'
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
DEFAULT_TEMPLATE = 'elsevier_template.tex'

# ========== IMAGE HANDLING ==========
def save_base64_figures(figures):
    """Save base64-encoded images to FIGURE_DIR and return updated figure dicts."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    saved = []
    for fig in figures:
        if 'base64' in fig and fig['base64']:
            img_data = base64.b64decode(fig['base64'])
            img_path = os.path.join(FIGURE_DIR, fig['filename'])
            with open(img_path, 'wb') as f:
                f.write(img_data)
        saved.append({**fig, 'base64': None})
    return saved

def save_uploaded_images(images):
    """Save Django InMemoryUploadedFile images to FIGURE_DIR and return figure dicts."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    figures = []
    for i, img in enumerate(images):
        img_path = os.path.join(FIGURE_DIR, img.name)
        with open(img_path, 'wb') as f:
            for chunk in img.chunks():
                f.write(chunk)
        figures.append({
            'filename': img.name,
            'page': None,
            'index': i + 1,
            'base64': None
        })
    return figures

def latex_escape(text):
    """Escape LaTeX special characters in user-supplied text."""
    if not isinstance(text, str):
        return text
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '_': r'\_',
        '%': r'\%',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text

def sanitize_metadata(metadata):
    """Ensure metadata.authors and affiliations are non-empty and valid for LaTeX."""
    # Remove authors with empty or None name
    authors = [a for a in metadata.get('authors', []) if a and a.get('name', '').strip()]
    if not authors:
        authors = [{"name": "Anonymous", "affiliation": 1}]
    # Remove empty or None affiliations
    affiliations = [a for a in metadata.get('affiliations', []) if a and str(a).strip()]
    if not affiliations:
        affiliations = ["Unknown Institution"]
    metadata['authors'] = authors
    metadata['affiliations'] = affiliations
    return metadata

# ========== LATEX RENDERING ==========
def render_latex(context, template_name=DEFAULT_TEMPLATE):
    """Render LaTeX template with provided context dict."""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    env.filters['latex_escape'] = latex_escape
    template = env.get_template(template_name)
    
    # Ensure sections have 'content' field for LaTeX template
    for section in context.get('sections', []):
        if 'text' in section and 'content' not in section:
            section['content'] = section['text']
    
    rendered_tex = template.render(**context)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tex_path = os.path.join(OUTPUT_DIR, 'generated_report.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(rendered_tex)
    return tex_path

# ========== PDF COMPILATION ==========
def compile_pdf(tex_path):
    """Compile LaTeX file to PDF. Returns True if successful, False otherwise. Logs output to outputs/latex_error.log."""
    cwd = os.path.dirname(tex_path)
    filename = os.path.basename(tex_path)
    # Always run pdflatex from the OUTPUT_DIR, and use only the filename
    cmd = [
        'pdflatex',
        '-interaction=nonstopmode',
        filename
    ]
    try:
        result = subprocess.run(cmd, cwd=OUTPUT_DIR, capture_output=True, text=True, timeout=60)
        log_path = os.path.join(OUTPUT_DIR, 'latex_error.log')
        with open(log_path, 'w', encoding='utf-8') as logf:
            logf.write('STDOUT:\n')
            logf.write(result.stdout)
            logf.write('\nSTDERR:\n')
            logf.write(result.stderr)
        print('LaTeX STDOUT:', result.stdout)
        print('LaTeX STDERR:', result.stderr)
        if result.returncode != 0:
            print(f'LaTeX compilation failed. See {log_path} for details.')
            return False
        return True
    except Exception as e:
        print('Exception during LaTeX compilation:', e)
        return False

def insert_figures_tables_charts(sections, figures, tables):
    def figure_latex(idx, fig):
        return (
            "\\begin{figure}[ht]"\
            "\n    \\centering"\
            f"\n    \\includegraphics[width=0.8\\textwidth]{{figures/{fig['filename']}}}"\
            f"\n    \\caption{{Figure {idx+1}}}"\
            "\n\\end{figure}\n"
        )
    def table_latex(idx, table):
        rows = table.get('data', table)
        ncols = max(len(r) for r in rows) if rows else 1
        latex = [
            "\\begin{table}[ht]",
            "    \\centering",
            f"    \\begin{{tabular}}{{{'l' * ncols}}}",
            "        \\toprule"
        ]
        if rows:
            # Header row
            header = rows[0] + [''] * (ncols - len(rows[0]))
            latex.append("        " + " & ".join(f"\\textbf{{{latex_escape(str(cell))}}}" for cell in header) + " \\\\")
            latex.append("        \\midrule")
            # Data rows
            for row in rows[1:]:
                padded = row + [''] * (ncols - len(row))
                latex.append("        " + " & ".join(latex_escape(str(cell)) for cell in padded) + " \\\\")
            latex.append("        \\bottomrule")
        latex += [
            "    \\end{tabular}",
            f"    \\caption{{Table {idx+1}}}",
            "    \\vspace{1em}",
            "\\end{table}"
        ]
        return "\n".join(latex)

    for i, section in enumerate(sections):
        def fig_repl(match):
            num = int(match.group(1)) - 1
            if 0 <= num < len(figures):
                return figure_latex(num, figures[num])
            return ''
        def table_repl(match):
            num = int(match.group(1)) - 1
            if 0 <= num < len(tables):
                return table_latex(num, tables[num])
            return ''
        content = section.get('content', section.get('text', ''))
        content = re.sub(r'\[FIGURE (\d+)\]', fig_repl, content)
        content = re.sub(r'\[TABLE (\d+)\]', table_repl, content)
        section['content'] = content
    return sections

# ========== ENTRY POINTS ==========
def generate_report(metadata, abstract, sections, keywords, figures, template_name=DEFAULT_TEMPLATE):
    """Generate PDF report from structured data and base64 figures."""
    figures = save_base64_figures(figures)
    metadata = sanitize_metadata(metadata)
    context = {
        'metadata': metadata,
        'abstract': abstract,
        'sections': sections,
        'keywords': keywords,
        'figures': figures
    }
    tex_path = render_latex(context, template_name)
    success = compile_pdf(tex_path)
    pdf_path = os.path.join(OUTPUT_DIR, 'generated_report.pdf')
    if not success or not os.path.exists(pdf_path):
        raise RuntimeError('PDF generation failed.')
    return pdf_path

def generate_report_from_form(metadata, abstract, sections, keywords, images, tables=None, template_name=DEFAULT_TEMPLATE):
    """Generate PDF report from Django form data (uploaded images)."""
    print('generate_report_from_form: tables:', tables)
    figures = save_uploaded_images(images)
    tables = tables or []
    metadata = sanitize_metadata(metadata)
    # Insert figures/tables into section content
    sections = insert_figures_tables_charts(sections, figures, tables)
    print('generate_report_from_form: sections after insert:', sections)
    context = {
        'metadata': metadata,
        'abstract': abstract,
        'sections': sections,
        'keywords': keywords,
        'figures': figures,
        'tables': tables
    }
    tex_path = render_latex(context, template_name)
    success = compile_pdf(tex_path)
    pdf_path = os.path.join(OUTPUT_DIR, 'generated_report.pdf')
    if not success or not os.path.exists(pdf_path):
        raise RuntimeError('PDF generation failed.')
    return pdf_path
