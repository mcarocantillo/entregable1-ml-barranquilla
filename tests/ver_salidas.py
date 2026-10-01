import sys, nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    for o in c.get("outputs", []):
        if o.output_type == "stream": txt = o.text
        elif o.output_type in ("execute_result", "display_data"):
            txt = o.data.get("text/plain", "") if "image/png" not in o.data else "[imagen]"
        elif o.output_type == "error": txt = "ERROR " + o.ename + ": " + o.evalue
        else: continue
        print(f"--- celda {i} ---"); print(txt[:int(sys.argv[2]) if len(sys.argv)>2 else 1500])
