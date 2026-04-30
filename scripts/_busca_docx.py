import zipfile, re

files = [
    "D:/UPE/Reconhecimento de padrões/Descrição de Projeto RECPAD/Projeto_RECPAD.docx",
    "D:/UPE/Reconhecimento de padrões/DOCUMENTO TÉCNICO RP.docx",
    "D:/UPE/Reconhecimento de padrões/Guia Completo.docx",
]

for fpath in files:
    try:
        with zipfile.ZipFile(fpath) as z:
            with z.open("word/document.xml") as f:
                content = f.read().decode("utf-8")
        text = re.sub(r"<[^>]+>", "", content)
        hits = [
            (i, line)
            for i, line in enumerate(text.splitlines(), 1)
            if "0,35" in line or "0.35" in line or "loss" in line.lower() or "Loss" in line
        ]
        print(f"=== {fpath} ===")
        if hits:
            for ln, line in hits:
                print(f"  linha {ln}: {line.strip()[:150]}")
        else:
            print("  (sem ocorrências)")
    except Exception as e:
        print(f"  ERRO: {e}")
