"""Hook S7 into supplement.tex (idempotent)."""
from pathlib import Path

p = Path(__file__).resolve().parent / "supplement.tex"
s = p.read_text(encoding="utf-8")
if "\\usepackage{longtable}" not in s:
    s = s.replace("\\usepackage{booktabs,array,graphicx}", "\\usepackage{booktabs,array,graphicx,longtable}")
if "\\input{supp_s7.tex}" not in s:
    s = s.replace("\\bibliographystyle{IEEEtran}", "\\input{supp_s7.tex}\n\n\\bibliographystyle{IEEEtran}")
s = s.replace("Section S6 records\nthe reconstruction provenance.",
              "Section S6 records\nthe reconstruction provenance, and Section S7 contains all results added in the\n"
              "first revision (IoT-76550-2026).")
p.write_text(s, encoding="utf-8")
print("patched", "\\input{supp_s7.tex}" in s, "longtable" in s)
