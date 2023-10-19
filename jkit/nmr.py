#!/usr/bin/env python
# coding:utf-8

#%%

import codecs
import re
import unicodedata
import warnings
from collections import defaultdict
from io import StringIO
from pathlib import Path

import nmrglue as ng
import numpy as np
import pandas as pd
import pynmrstar
import scipy.sparse
from . import aminoacids
from .util import chunked_iterator
from joblib import Parallel, delayed
import os
from tqdm import tqdm

gamma = {
    "H1": 1.0,
    "H2": 0.153506088,
    "C13": 0.251449530,
    "N15": 0.101329118,
    "P31": 0.404808636,
    "F19": 0.940746805,
}


def parse_procpar(path="./procpar"):
    def num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    procpar = defaultdict(lambda: None)
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            if line[0][0].isalpha():
                fieldname = line[0]
                line = f.readline()
                tok = line.split()
                l = num(tok[0])
                if l == 1 and len(tok) == 2:
                    if tok[1][0] == '"':
                        procpar[fieldname] = tok[1].strip('"\n')
                    else:
                        procpar[fieldname] = num(tok[1])
                elif len(tok) > 2 and tok[1][0] != '"':
                    ary_vals = [num(s) for s in tok]
                    if len(ary_vals) != l + 1:
                        print(
                            "Warning: reported array length and actual array length do not match~"
                        )
                    procpar[fieldname] = np.array(ary_vals[1:])
                else:
                    procpar[fieldname] = [line.lstrip("0123456789 ").strip('"\n')]
                    for i in range(1, l):
                        line = f.readline()
                        procpar[fieldname].append(line.strip('"\n'))
    return procpar


def parse_sparky_peak_name(s):
    atom_regexp = "([HFPCN\?]\w*\d?)"
    resid_regexp = "([A-Z])(\d+)(\w*)"
    peak_name_regexp = "((?:(?:(?:[A-Z])(?:\d+)(?:\w*))?(?:[HFPCN]\w*\d?.*)\W?){2,3})"
    pknm = re.search(peak_name_regexp, s)
    if not pknm is None:
        pk = re.findall("(?:{})?{}".format(resid_regexp, atom_regexp), pknm.group(0))
        for i, atm in enumerate(pk):
            pk[i] = list(pk[i])
            if atm[0] == "" and atm[3] != "?":
                pk[i][0] = pk[i - 1][0]
                pk[i][1] = pk[i - 1][1]
            try:
                pk[i][1] = int(pk[i][1])
            except (ValueError, TypeError):
                pk[i][1] = None
        return tuple([tuple(e) for e in pk])
    else:
        return None


def read_sparky_table(path):
    with open(path, "r") as f:
        header = f.readline()
        col_names = [e.strip() for e in re.split(r"\s{2,}", header) if len(e) > 0]
        col_indices = [0 for e in col_names]
        for i, cn in enumerate(col_names):
            idx = header.index(cn)
            col_indices[i] = idx + len(cn) + 4

        col_slices = [
            slice(col_indices[i - 1], col_indices[i])
            if i > 0
            else slice(0, col_indices[0])
            for i in range(len(col_indices))
        ]

        table = []
        for l in f.readlines():
            if len(l) > 1:
                table.append([l[slc].strip() for slc in col_slices])

    df = pd.DataFrame(table, columns=col_names)
    df = df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
    return df


def sparky_reslist_to_dataframe(path):
    try:
        df = read_sparky_table(path)
        atoms = [None] * len(df.index)
        for i in range(len(df.index)):
            restyp, resnum = re.match("([A-Z])(\d+)", df["Group"][i]).groups()
            atoms[i] = (restyp, resnum, df["Atom"][i])
        df.drop(["Group", "Atom", "Nuc", "Assignments"], axis=1, inplace=True)
        df.loc[:, "residue_type"], df.loc[:, "residue_number"], df.loc[:, "atom"] = zip(
            *atoms
        )
        df.rename(columns={"Shift": "w", "SDev": "werr"}, inplace=True)
        df = df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
        df = df.dropna()
        return df
    except IOError:
        print("cannot open %s" % path)
        return None


class SparkyParameter(object):
    def __init__(self, s):
        if s.split(b" ")[0] in [b"set", b"default"]:
            if len(s.split(b" ")[1].split(b".")) > 1:
                self.name = s.split(b" ")[1].split(b".")
            else:
                self.name = s.split(b" ")[1]
            self.value = s.split(b" ")[2:]
        else:
            if len(s.split(b" ")) > 1 and len(s.split(b" ")[0].split(b".")) > 1:
                self.name = s.split(b" ")[0].split(b".")
            else:
                self.name = s.split(b" ")[0]
            self.value = s.split(b" ")[1:]
        self.children = []


class SparkySection(object):
    def __init__(self, tag, contents):
        self.tag = tag
        self.contents = contents
        self.children = []
        self.child_strings = []
        self.parameters = []
        str_list = re.split(b"]\s+type", self.contents)
        for i, e in enumerate(str_list):
            s1 = b"type" + e + (b"]" if i < len(str_list) - 1 else b"")
            cur_par = None
            for s2 in s1.split(b"\n"):
                if len(s2) > 0:
                    if s2[0] == b"[":
                        cur_par = len(self.parameters) - 1
                    elif s2[0] == b"]":
                        cur_par = None
                    else:
                        if cur_par is None:
                            self.parameters.append(SparkyParameter(s2))
                        else:
                            self.parameters[cur_par].children.append(
                                SparkyParameter(s2)
                            )

    def __repr__(self):
        return f"<SparkySection {self.tag}>"

    __str__ = __repr__


class SparkyPeak(object):
    def __init__(self, element, ndim, group=None):
        typ = element.get(b"type", None)
        if len(typ) > 0 and typ[0] != b"peak":
            raise ValueError("element data is not for a peak")

        _id = element.get(b"id", None)
        self.id = int(_id[0]) if len(_id) > 0 else None

        _pos = element.get(b"pos", None)
        self.pos = tuple(float(e) for e in _pos) if len(_pos) > 0 else None
        if len(self.pos) != ndim:
            raise ValueError("peak position is not the correct dimensionality")

        _height = element.get(b"height", None)
        self.height = float(_height[-1]) if len(_height) > 0 else None

        self.group = group
        self.assignment = ((None,) * 4,) * ndim

    @property
    def ndim(self):
        return len(self.pos)

    def assign(self, assignment):
        try:
            if len(assignment) == self.ndim and all(len(e) == 4 for e in assignment):
                self.assignment = assignment
        except TypeError:
            pass

    def __repr__(self):
        assignment_str = "-".join(
            [
                "".join((str(e1) if e1 is not None else "?") for e1 in e0)
                for e0 in self.assignment
            ]
        )

        if self.group is not None:
            return f"<SparkyPeak assignment = {assignment_str} group = {self.group} id = {self.id} pos = {self.pos} height = {self.height}>"
        else:
            return f"<SparkyPeak assignment = {assignment_str} id = {self.id} pos = {self.pos} height = {self.height}>"


class SparkySpectrum(object):
    def __init__(
        self,
        path,
        temperature=0,
        pressure=1,
        pH=7.0,
        concentration=1.0,
        sample_id=0,
        offset=None,
    ):
        self.name = ""
        self.ucsf = ""
        self.ndim = 0
        self.shift = (0.0,)
        self.shape = (0,)

        self.path = path
        self.temperature = temperature
        self.pressure = pressure
        self.pH = pH
        self.concentration = concentration
        self.sample_id = sample_id

        with codecs.open(self.path, encoding="utf-8") as f:
            file_str = f.read()
        file_str = unicodedata.normalize("NFKD", file_str).encode("ascii", "ignore")

        section_regexp = br"<(?P<tag_name>.+)>(?P<contents>.*?)<end (?P=tag_name)>"
        ornament_regexp = br"\[(?P<contents>.*?)\]"

        sections = []

        queue = [(0, None, file_str)]
        while queue:
            i, parent_idx, s = queue.pop(0)
            for m in re.finditer(section_regexp, s, re.DOTALL):
                if m is not None:
                    sec = SparkySection(m.group("tag_name"), m.group("contents"))
                    if parent_idx is not None:
                        sections[parent_idx].children.append(sec)
                    sections.append(sec)
                    queue.append((i + 1, len(sections) - 1, m.group("contents")))

        def split_peaks(seq):
            group = []
            for p in seq:
                if p.name == b"type" and len(p.value) > 0:
                    yield group
                    group = []
                    group.append(p)
                else:
                    group.append(p)
            yield group

        for sect in sections:
            if sect.tag == b"spectrum":
                for par in sect.parameters:
                    if par.name == b"name" and len(par.value) > 0:
                        self.name = par.value[0]
                    elif par.name == b"pathname" and len(par.value) > 0:
                        self.ucsf = Path(par.value[0].decode("utf-8")).resolve()
                    elif par.name == b"dimension" and len(par.value) > 0:
                        self.ndim = int(par.value[0])
                    elif par.name == b"shift" and len(par.value) > 0:
                        if offset is not None:
                            self.shift = tuple(
                                float(e) + offset[i] for i, e in enumerate(par.value)
                            )
                        else:
                            self.shift = tuple(
                                float(e) for i, e in enumerate(par.value)
                            )
                    elif par.name == b"points" and len(par.value) > 0:
                        self.shape = tuple(int(e) for e in par.value)

        elements = []
        for e in sections:
            if e.tag == b"ornament":
                for pk in split_peaks(e.parameters):
                    elements.append({e.name: e.value for e in pk})

        peak_groups = defaultdict(list)
        cur_pg = None
        for e in elements:
            if len(e[b"type"]) > 0 and e[b"type"][0] == b"peakgroup":
                cur_pg = int(e[b"id"][0])
            elif len(e[b"type"]) > 0 and e[b"type"][0] == b"peak":
                peak_groups[cur_pg].append(SparkyPeak(e, ndim=self.ndim, group=cur_pg))
            else:
                if (
                    b"type" in e.keys()
                    and b"label" in e[b"type"]
                    and b"pos" in e.keys()
                ):
                    for e1 in e.get(b"label", b"?"):
                        if cur_pg is not None:
                            for pk in peak_groups[cur_pg]:
                                pk.assign(parse_sparky_peak_name(str(e1)))
                        else:
                            peak_groups[cur_pg][-1].assign(
                                parse_sparky_peak_name(str(e1))
                            )

        self.peaks = [e1 for e0 in peak_groups.values() for e1 in e0]
        self.ucsf_initialized = False

    def peak_dataframe(self, update_peak_heights=True):
        peak_list = [
            sum(
                [
                    [
                        pk.assignment[i][0],
                        pk.assignment[i][1],
                        pk.assignment[i][2],
                        pk.assignment[i][3],
                    ]
                    for i in range(self.ndim)
                ],
                [],
            )
            + [pk.group]
            + list(pk.pos)
            + [pk.height]
            for pk in self.peaks
        ]
        peak_df = pd.DataFrame(
            peak_list,
            columns=sum(
                [
                    [
                        "residue_type_{:d}".format(i + 1),
                        "residue_number_{:d}".format(i + 1),
                        "residue_mod_{:d}".format(i + 1),
                        "atom_{:d}".format(i + 1),
                    ]
                    for i in range(self.ndim)
                ],
                [],
            )
            + ["group"]
            + ["w_{:d}".format(i + 1) for i in range(self.ndim)]
            + ["height"],
        )
        if update_peak_heights:
            peak_df.loc[:, "new_height"] = [
                self.get_peak_intensity(list(pk.pos)) for pk in tqdm(self.peaks)
            ]
        return peak_df

    def initialize_ucsf(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ucsf_dic, self.ucsf_data = ng.sparky.read_lowmem(self.ucsf)

        # make ppm scales
        self.uc = []
        self.ppm_scales = []
        self.ppm_limits = []
        for i in range(self.ndim):
            self.uc.append(ng.sparky.make_uc(self.ucsf_dic, self.ucsf_data, dim=i))
            self.ppm_scales.append(self.uc[-1].ppm_scale())
            self.ppm_limits.append(self.uc[-1].ppm_limits())

        idx_sel = zip(
            *[
                np.random.randint(0, self.ucsf_data.shape[i], size=100)
                for i in range(self.ndim)
            ]
        )
        data_sel = np.array([self.ucsf_data[idx] for idx in idx_sel])
        sigma = 1.4826 * np.median(np.abs(data_sel - np.median(data_sel)))
        mu = np.median(data_sel[np.abs(data_sel) < 6 * sigma])
        data_sel = data_sel[np.abs(data_sel - mu) < 6 * sigma]
        self.sigma = 1.4826 * np.median(np.abs(data_sel - np.median(data_sel)))

    def get_peak_intensity(self, pos):
        return self.get_region(pos)[0].item()

    def get_region(self, limits):
        if not self.ucsf_initialized:
            self.initialize_ucsf()

        slc_list = []
        ppm_limits = []
        for i in range(self.ndim):
            if limits[i] is None:
                slc_list.append(slice(None))
                ppm_limits.append(
                    (
                        self.uc[i].ppm(0) + self.shift[i],
                        self.uc[i].ppm(self.ucsf_data.shape[i]) + self.shift[i],
                    )
                )
            elif not hasattr(limits[i], "__iter__"):
                idx = self.uc[i](f"{limits[i]-self.shift[i]} ppm")
                slc_list.append(idx)
                ppm_limits.append(self.uc[i].ppm(idx) + self.shift[i])
            else:
                idx = sorted(
                    [
                        min(
                            self.ucsf_data.shape[i],
                            max(0, self.uc[i](f"{limits[i][0] - self.shift[i]} ppm")),
                        ),
                        min(
                            self.ucsf_data.shape[i],
                            max(0, self.uc[i](f"{limits[i][1] - self.shift[i]} ppm")),
                        ),
                    ]
                )
                ppm_limits.append(
                    (
                        self.uc[i].ppm(idx[0]) + self.shift[i],
                        self.uc[i].ppm(idx[1]) + self.shift[i],
                    )
                )
                slc_list.append(slice(*idx))
        return self.ucsf_data[tuple(slc_list)], ppm_limits

    def rename_atoms(self, name_dict):
        renamed_peaks = []
        for pk in self.peaks:
            for i in range(pk.ndim):
                if pk.assignment[i][3] in name_dict.keys():
                    pk.assignment[i] = (
                        pk.assignment[i][0],
                        pk.assignment[i][1],
                        pk.assignment[i][2],
                        name_dict[pk.assignment[i][0]],
                    )
            renamed_peaks.append(pk)
        self.peaks = renamed_peaks

    @property
    def resonance_list(self):
        return self._calculate_resonance_list()

    def _calculate_resonance_list(self, dimensions=None):
        dimensions = dimensions or list(range(self.ndim))
        resonances = []
        for pk in self.peaks:
            for i in dimensions:
                if (
                    pk.assignment[i][0] is not None
                    and pk.assignment[i][1] is not None
                    and pk.assignment[i][3] is not None
                ):
                    resonances.append(
                        [
                            pk.assignment[i][0],
                            pk.assignment[i][1],
                            pk.assignment[i][3],
                            pk.pos[i],
                            pk.height,
                        ]
                    )

        df = pd.DataFrame(
            resonances,
            columns=["residue_type", "residue_num", "atom", "chemical_shift", "height"],
        )
        resonance_list = (
            df.groupby(["residue_type", "residue_num", "atom"])["chemical_shift"]
            .agg(
                [
                    (
                        "resonance",
                        lambda x: np.average(x, weights=df.loc[x.index, "height"]),
                    ),
                    (
                        "error",
                        lambda x: np.sqrt(
                            np.average(
                                np.abs(
                                    x - np.average(x, weights=df.loc[x.index, "height"])
                                )
                                ** 2,
                                weights=df.loc[x.index, "height"],
                            )
                        ),
                    ),
                ]
            )
            .reset_index()
        )
        return resonance_list


def read_sparky_peaklist(path, return_dataframe=False):
    try:
        df = read_sparky_table(path)
        df.loc[:, "Assignment"] = df["Assignment"].apply(parse_sparky_peak_name)
        return df
    except IOError:
        print("cannot open %s" % path)
        if return_dataframe:
            return None, None
        else:
            return None


def normalize_sparky_peaklist(df, force_use_volume=False):
    nd = (
        6
        if "w6" in df.columns
        else (
            5
            if "w5" in df.columns
            else (
                4
                if "w4" in df.columns
                else (
                    3
                    if "w3" in df.columns
                    else (
                        2 if "w2" in df.columns else (1 if "w1" in df.columns else None)
                    )
                )
            )
        )
    )

    if nd is None:
        raise ValueError

    split_atoms = []
    for e in df["Assignment"]:
        if e is None:
            split_atoms.append([None] * nd)
        else:
            split_atoms.append(list(e) + [None] * (nd - len(e)))

    for i, c in enumerate(zip(*split_atoms)):
        df.loc[:, "atom_{}".format(i + 1)] = c

    df.drop("Assignment", axis=1, inplace=True)
    if force_use_volume and "Volume" in df.columns:
        intensity = -np.ones(len(df.index))
        for i, e in enumerate(df["Volume"]):
            if len(e.strip()) > 0:
                intensity[i] = float(e.strip().split()[0])
        df.loc[:, "intensity"] = intensity
    elif "Data Height" in df.columns:
        df.loc[:, "intensity"] = df["Data Height"]
    elif "S/N" in df.columns:
        df.loc[:, "intensity"] = df["S/N"]
    elif "Fit height" in df.columns:
        df.loc[:, "intensity"] = df["Fit height"]
    elif "Volume" in df.columns:
        intensity = -np.ones(len(df.index))
        for i, e in enumerate(df["Volume"]):
            if len(e.strip()) > 0:
                intensity[i] = float(e.strip().split()[0])
        df.loc[:, "intensity"] = intensity

    for i in range(nd):
        if "lw{} (hz)".format(i + 1) in df.columns:
            df.loc[:, "werr{}".format(i + 1)] = df["lw{} (hz)".format(i + 1)]
        else:
            df.loc[:, "werr{}".format(i + 1)] = np.zeros(len(df.index))

    for c in df.columns:
        if c not in ["intensity"] + ["werr{}".format(i + 1) for i in range(nd)] + [
            "w{}".format(i + 1) for i in range(nd)
        ] + ["atom_{}".format(i + 1) for i in range(nd)]:
            df.drop(c, axis=1, inplace=True)

    return df


def sparky_peaklist_to_dataframe(path, force_use_volume=False):
    df = read_sparky_peaklist(path)
    df = normalize_sparky_peaklist(df, force_use_volume=force_use_volume)
    return df


def peak_dataframe_to_sparky_peaklist(df, path, dim_order=None):
    ndim = len([n for n in df.columns if n[:5] == "atom_name_"])
    if dim_order is None:
        dim_order = tuple(range(ndim))
    inv_dim_order = {v: i for i, v in enumerate(dim_order)}

    header_str = f'{"Assignment":>16s}'
    header_str += "".join(f"         w{i}" for i in range(1, ndim + 1))

    with open(path, "w") as f:
        f.write(header_str + "\n\n")
        for i, pk in df.iterrows():
            cur_res = None
            pk_str_parts = []
            for j in range(ndim):
                idx = dim_order[j]
                residue_typ = pk[f"residue_type_{idx}"]
                residue_num = pk[f"residue_number_{idx}"]
                atom = pk[f"atom_name_{idx}"]
                condition = ""
                if cur_res != (residue_num, residue_typ):
                    pk_str_parts.append(
                        residue_typ + str(residue_num) + str(condition) + atom
                    )
                    cur_res = (residue_num, residue_typ)
                else:
                    pk_str_parts.append(atom)

            line = f'{"-".join(pk_str_parts):>17s}'
            for j in range(ndim):
                idx = dim_order[j] + 1
                line += f'{pk[f"w_{idx}"]:>11.3f}'
            f.write(line + " \n")


def flya_reslist_to_dataframe(path, seq):
    try:
        df = pd.read_table(
            path,
            header=None,
            delim_whitespace=True,
            names=["index", "w", "werr", "atom", "residue_number"],
        )

        df.loc[:, "residue_type"] = [seq[i - 1] for i in df["residue_number"]]
        df.drop("index", axis=1, inplace=True)
        df = df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
        return df
    except IOError:
        print("cannot open %s" % path)
        return None


def read_flya_peaklist(path, seq, return_dataframe=False):
    def parse_atom(s):
        try:
            e1, e2 = s.split(".")
            atom_type, residue_number = e1, int(e2)
            return seq[residue_number - 1], residue_number, atom_type
        except:
            return float("NaN")

    try:
        with open(path, "r") as f:
            first_line = f.readline()
            nd = None
            fmt_nd = None
            md = re.match(r"# Number of dimensions (\d+)", first_line)
            if md:
                nd = int(md.group(1))
            second_line = f.readline()
            md = re.match(r"#FORMAT xeasy(\d+)D", second_line)
            if md:
                fmt_nd = int(md.group(1))
            if (nd is None and fmt_nd is None) or nd != fmt_nd:
                raise ValueError
            else:
                nd = nd if not nd is None else fmt_nd
            table = [ln.split("#")[0].split() for ln in f.readlines() if ln[0] != "#"]
        column_names = (
            ["index"]
            + ["w{}".format(i) for i in range(nd)]
            + ["color", "spectype", "volume", "volume_error", "integration_mode", "?"]
            + ["atom_{}".format(i) for i in range(nd)]
        )
        converters = {"atom_{}".format(i): parse_atom for i in range(nd)}

        df = pd.DataFrame(table, columns=column_names)
        for i in range(nd):
            df.loc[:, "atom_{}".format(i)] = df["atom_{}".format(i)].apply(parse_atom)

        df.drop("?", axis=1, inplace=True)
        df = df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
        return df
    except IOError:
        print("cannot open %s" % path)
        return None


def normalize_flya_peaklist(df):
    nd = (
        6
        if "w6" in df.columns
        else (
            5
            if "w5" in df.columns
            else (
                4
                if "w4" in df.columns
                else (
                    3
                    if "w3" in df.columns
                    else (
                        2 if "w2" in df.columns else (1 if "w1" in df.columns else None)
                    )
                )
            )
        )
    )

    if nd is None:
        raise ValueError

    df.loc[:, "intensity"] = df["volume"]

    for i in range(nd):
        df.loc[:, "werr{}".format(i + 1)] = np.zeros(len(df.index))

    for c in df.columns:
        if c not in ["intensity"] + ["werr{}".format(i + 1) for i in range(nd)] + [
            "w{}".format(i + 1) for i in range(nd)
        ] + ["atom_{}".format(i + 1) for i in range(nd)]:
            df.drop(c, axis=1, inplace=True)

    return df


def flya_peaklist_to_dataframe(path, seq):
    df = read_flya_peaklist(path, seq)
    return normalize_flya_peaklist(df)


def bmrb_to_dataframe(bmrb_id):
    file_name = f"{bmrb_id}.bmrb.pkl"
    if os.path.isfile(file_name):
        df = pd.read_pickle(file_name)
    else:
        entry = pynmrstar.Entry.from_database(bmrb_id)
        shift_frames = entry.get_loops_by_category("Atom_chem_shift")
        shift_csv = shift_frames[0].get_data_as_csv()
        df = pd.read_csv(StringIO(shift_csv))
        df.loc[:, "residue_type"] = [
            aminoacids.three_to_one[e] for e in df["_Atom_chem_shift.Comp_ID"]
        ]
        df.index.name = None
        df.rename(
            columns={
                "_Atom_chem_shift.Val": "w",
                "_Atom_chem_shift.Val_err": "w_err",
                "_Atom_chem_shift.Seq_ID": "residue_number",
                "_Atom_chem_shift.Atom_ID": "atom",
            },
            inplace=True,
        )
        df = df.loc[:, ["residue_number", "residue_type", "atom", "w", "w_err"]]
        df.to_pickle(file_name)
    return df.reset_index(drop=True)


def estimate_spectrum_noise(data, sigma_0, chunk_shape, overlap):
    # estimate spectrum noise
    # procedure:
    # divide the spectrum into overlapping chunks
    # for each chunk:
    # - measure the median absolute deviation from the median of intensities
    # - measure the median intensity
    # - measure the skew of the intensities (high if the chunk contains peaks of one sign)
    # - measure the kurtosis of the intensities (high if the chunk contains peaks of both signs)
    #
    data = np.atleast_2d(data)
    filt_data = np.ma.masked_array(data, mask=(np.abs(data) >= 5 * sigma_0))
    n1, n2 = filt_data.shape
    mus = []
    sigmas = []
    medcouples = []
    for pos, chnk in chunked_iterator(filt_data, chunk_shape, overlap):
        if not chnk.mask.all():
            x = chnk.reshape(chnk.size)
            if jarque_bera(x.reshape(-1))[0] < 5.99 * np.exp(-6 * t ** (-0.85)):
                sigmas.append(mad)
    return np.median(sigmas)


def baseline_als(y, lam, p, niter, eps=1e-9):
    l = len(y)
    d = scipy.sparse.csc_matrix(np.diff(np.eye(l), 2))
    w = np.ones(l)
    z_prev = np.inf * np.ones(y.shape)
    for i in range(niter):
        W = scipy.sparse.spdiags(w, 0, l, l)
        Z = W + lam * d.dot(d.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
        if np.max(np.abs(z_prev - z)) / np.max(np.abs(y)) < eps:
            break
        z_prev = z[:]
    return z


def nd_baseline(data, lam, p, niter, dims):
    baseline = np.zeros(list(data.shape) + [data.ndim])
    if data.ndim == 1:
        return baseline_als(data, lam, p, niter)
    else:
        if dims is None:
            dims = range(data.ndim)
        for d in dims:
            baseline[..., d] = np.transpose(
                (
                    np.array(
                        Parallel(n_jobs=8)(
                            delayed(baseline_als)(vec, lam=1e1, p=0.2, niter=10)
                            for vec in np.transpose(
                                data, np.roll(range(data.ndim), d)
                            ).reshape(-1, data.shape[d])
                        )
                    )
                ).reshape(np.roll(data.shape, d)),
                np.roll(range(data.ndim), -d),
            )
        for d in range(1, data.ndim):
            idx = np.where(np.abs(baseline[:, :, 0]) > np.abs(baseline[:, :, d]))
            baseline[idx[0], idx[1], 0] = baseline[idx[0], idx[1], d]
        return np.mean(baseline, axis=-1)


def fit_baseline(data, lam=1e1, p=0.2, niter=10, verbose=False, dims=None):
    sigma_0 = 1.4826 * np.median(np.abs(data - np.median(data)))
    if verbose:
        print("Rough noise estimate: {:6.3f}".format(sigma_0))
    if verbose:
        print("Fitting Initial baseline")
    bl = nd_baseline(data, lam, p, niter, dims)
    m_0 = np.abs(data - bl) >= 5 * sigma_0
    filt_data_0 = np.ma.masked_array(data, mask=m_0)
    sigma = estimate_spectrum_noise(filt_data_0, sigma_0, (32, 32), (28, 28))
    if verbose:
        print("Refined noise estimate: {:6.3f}".format(sigma))
    if verbose:
        print("Fitting Refined baseline")
    filt_bl_0 = nd_baseline(filt_data_0, lam, p, niter, dims)
    m_1 = np.abs(data - filt_bl_0) >= 5 * sigma
    filt_data_1 = np.ma.masked_array(data, mask=m_1)
    if verbose:
        print("Fitting Final baseline")
    filt_bl_1 = nd_baseline(filt_data_1, lam, p, niter, dims)
    if verbose:
        print("done")

    fixed_data = data - filt_bl_1
    baseline = filt_bl_1

    return fixed_data, baseline, sigma
