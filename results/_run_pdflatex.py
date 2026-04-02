import subprocess, os, sys

cwd = r'd:\Naser\2\Journal Paper\journal_v2'
out_file = r'd:\Naser\2\results\_pdftex_run.txt'

print("Running pdflatex...")
r = subprocess.run(
    ['pdflatex', '-interaction=nonstopmode', 'sn-article.tex'],
    capture_output=True, text=True, encoding='utf-8', errors='replace',
    cwd=cwd
)

with open(out_file, 'w', encoding='utf-8') as f:
    f.write(r.stdout)
    f.write('\n--- STDERR ---\n')
    f.write(r.stderr)

errs = [l.strip() for l in r.stdout.split('\n') if l.startswith('!')]
print(f"Return code: {r.returncode}")
print(f"Errors ({len(errs)}):")
for e in errs:
    print(" ", e)

pdf_path = os.path.join(cwd, 'sn-article.pdf')
if os.path.exists(pdf_path):
    sz = os.path.getsize(pdf_path)
    print(f"PDF exists: {sz} bytes")
else:
    print("PDF NOT found")

print(f"Full output saved to: {out_file}")
