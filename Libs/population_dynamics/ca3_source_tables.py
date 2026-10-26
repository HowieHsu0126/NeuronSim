"""Read the two original CA3 source supplements without evaluating formulas."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .provenance import sha256_file

WORD = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
SHEET = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RELATIONSHIP_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def read_docx(path: Path) -> dict:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        tables = []
        for table in root.findall(".//w:tbl", WORD):
            tables.append([
                [" ".join(t.text or "" for t in cell.findall(".//w:t", WORD))
                 for cell in row.findall("w:tc", WORD)]
                for row in table.findall("w:tr", WORD)
            ])
        paragraphs = [" ".join(t.text or "" for t in p.findall(".//w:t", WORD))
                      for p in root.findall(".//w:p", WORD)]
        return {"tables": tables, "paragraphs": paragraphs,
                "members": archive.namelist()}


def read_xlsx(raw: bytes) -> dict:
    with ZipFile(BytesIO(raw)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", SHEET):
                strings.append("".join(t.text or "" for t in item.findall(".//s:t", SHEET)))
        relationships = {
            element.attrib["Id"]: element.attrib["Target"]
            for element in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        sheets = []
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in workbook.findall("s:sheets/s:sheet", SHEET):
            target = relationships[sheet.attrib[RELATIONSHIP_ID]]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            root = ET.fromstring(archive.read(member))
            cells = []
            for cell in root.findall("s:sheetData/s:row/s:c", SHEET):
                value_element = cell.find("s:v", SHEET)
                value = value_element.text if value_element is not None else None
                kind = cell.get("t", "n")
                if kind == "s" and value is not None:
                    value = strings[int(value)]
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//s:t", SHEET))
                formula = cell.find("s:f", SHEET)
                if value is not None or formula is not None:
                    entry = {"cell": cell.attrib["r"], "type": kind, "stored_value": value}
                    if formula is not None:
                        entry["formula"] = {"text": formula.text, "attributes": formula.attrib}
                    cells.append(entry)
            sheets.append({"name": sheet.attrib["name"], "member": member, "cells": cells,
                           "merged_cells": [m.attrib["ref"] for m in root.findall("s:mergeCells/s:mergeCell", SHEET)]})
        return {"sheets": sheets, "formulas_evaluated": False,
                "value_semantics": "Source XML values are strings; formula caches are read as stored, not recalculated."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.census) as archive:
        members = [{"name": info.filename, "bytes": info.file_size} for info in archive.infolist()]
        workbooks = {name: read_xlsx(archive.read(name))
                     for name in archive.namelist() if name.lower().endswith(".xlsx")}
        source_text = {name: archive.read(name).decode("utf-8-sig")
                       for name in archive.namelist() if name.lower().endswith((".r", ".txt"))}
    if not workbooks:
        raise ValueError("CENSUS_SUPPLEMENT_HAS_NO_XLSX_WORKBOOK: " + repr(members))
    result = {
        "status": "SOURCE_SUPPLEMENT_TABLES_READ",
        "sources": {"census": {"path": str(args.census), "sha256": sha256_file(args.census)},
                    "anatomy": {"path": str(args.anatomy), "sha256": sha256_file(args.anatomy)}},
        "census_members": members, "census_workbooks": workbooks,
        "census_source_text": source_text,
        "anatomy": read_docx(args.anatomy),
        "scientific_scale_qualified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "workbooks": {
        name: [s["name"] for s in book["sheets"]] for name, book in workbooks.items()},
        "anatomy_tables": len(result["anatomy"]["tables"])}))


if __name__ == "__main__":
    main()
