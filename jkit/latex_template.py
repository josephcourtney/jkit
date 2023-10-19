#!/usr/bin/env python

import string
import re
import warnings
import os
import subprocess
import shutil
import tempfile
from pathlib import Path


class LatexTemplate(object):
    def __init__(self, path, delimiter="#"):
        self.custom_template = type(
            "CustomTemplate", (string.Template,), {"delimiter": delimiter}
        )

        self.template_path = path
        with open(self.template_path, "r") as f:
            self.template_string = f.read()
        self.fields = set()
        self.substitutions = dict()
        self.template = self.custom_template(self.template_string)
        mo = re.findall(self.template.pattern, self.template.template)
        for _, e1, e2, _ in mo:
            e = e2 if len(e1) == 0 else e1
            self.fields.add(e)
            self.substitutions[e] = None
        self.open_fields = self.fields.copy()
        self.closed_fields = set()

    def set_field(self, field, value):
        if field in self.open_fields:
            self.substitutions[field] = value
        else:
            warnings.warn(
                f"Desired field, {repr(field)}, is closed with value, {repr(self.substitutions[field])}."
            )

    def substitute(self):
        kwargs = {
            f: self.substitutions[f]
            for f in self.open_fields
            if self.substitutions[f] is not None
        }
        self.template_string = self.template.safe_substitute(**kwargs)
        self.template = self.custom_template(self.template_string)
        self.closed_fields = self.closed_fields | set(kwargs.keys())
        self.open_fields -= set(kwargs.keys())

    def compile(self, out_path, safe=True):
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)

            self.substitute()
            if safe and len(self.open_fields) > 0:
                raise RuntimeError(
                    f"Fields {repr(self.open_fields)} remain open! Abort!"
                )
            with open(tmp_path / "compiled.tex", "w") as f:
                f.write(self.template_string)

            p = subprocess.Popen(
                ["pdflatex", "compiled.tex", "-halt-on-error"],
                cwd=tmp_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                p.wait(10)
                p.kill()
                if p.returncode == 0:
                    shutil.copyfile(tmp_path / "compiled.pdf", out_path)
                    return True
                else:
                    return False
            except subprocess.TimeoutExpired:
                return False
