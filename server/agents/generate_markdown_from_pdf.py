import pymupdf4llm

md_text = pymupdf4llm.to_markdown("data/pai/lectures/main.pdf")

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())