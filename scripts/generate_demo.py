import os

demo_dir = "demo"
assets_dir = os.path.join(demo_dir, "assets")
os.makedirs(assets_dir, exist_ok=True)

html_path = os.path.join(demo_dir, "index.html")

with open(html_path, "w") as f:
    f.write("<html><body><h1>OpenLLM OCR Annotator Demo</h1>\n")
    for img in os.listdir(assets_dir):
        f.write(f"<div><img src='assets/{img}' style='width:600px'><br>{img}</div><hr>\n")
    f.write("</body></html>")