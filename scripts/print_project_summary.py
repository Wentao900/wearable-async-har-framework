from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

print("wearable-async-har-framework")
print("-")
print("Docs:")
for p in sorted((ROOT / "docs").glob("*.md")):
    print(f"  - {p.name}")
print("Code skeleton:")
for p in sorted((ROOT / "src").rglob("*.py")):
    print(f"  - {p.relative_to(ROOT)}")
