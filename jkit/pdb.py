import os
import sys
import tempfile
from io import StringIO
from itertools import combinations
from sys import stdout
from urllib.request import Request, urlopen

import numpy as np

try:
    import openmm as mm
    import openmm.app as app
    import unit as unit
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer
except ModuleNotFoundError:
    pass
import pandas as pd
import scipy.spatial

from . import aminoacids

if "_line_profiler" not in sys.modules:

    def profile(f):
        return f


class PDBModel(object):
    # @profile
    def __init__(self, df, no_kd=False):
        self.df = df.sort_values(["res_num", "atom_name"])
        if not no_kd:
            self.build_kdtree()
        self._dmat = None
        self._bonds = None

    # @profile
    def build_kdtree(self):
        xyz_df = self.df[["x", "y", "z"]].dropna()
        self.kdtree_idx = self.df.dropna(subset=["x", "y", "z"]).index.values
        self.kdtree = scipy.spatial.KDTree(xyz_df.values)

    # @profile
    def distance_between_atoms(self, res_name_1, res_num_1, atom_name_1, res_name_2, res_num_2, atom_name_2):
        idx_1 = self.df[
            (self.df["res_name"] == res_name_1)
            & (self.df["res_num"] == res_num_1)
            & (self.df["atom_name"] == atom_name_1)
        ].index.tolist()
        idx_2 = self.df[
            (self.df["res_name"] == res_name_2)
            & (self.df["res_num"] == res_num_2)
            & (self.df["atom_name"] == atom_name_2)
        ].index.tolist()
        r = np.sqrt(np.sum((np.array(self.kdtree.data[idx_1[0]]) - np.array(self.kdtree.data[idx_2[0]])) ** 2))
        return r

    # @profile
    def nearest_atoms(self, res_name, res_num, atom_name, n):
        idx = self.df[
            (self.df["res_name"] == aminoacids.one_to_three[res_name])
            & (self.df["res_num"] == res_num)
            & (self.df["atom_name"] == atom_name)
        ].index.tolist()
        distances, near_idx = self.kdtree.query(self.kdtree.data[idx[0]], n)
        return [self.df.iloc[i] for i in near_idx], distances

    # @profile
    def atoms_within_distance(self, res_name, res_num, atom_name, r):
        if r == np.inf:
            return [self.df.iloc[i] for i in range9len(self.df.index)]
        else:
            idx = self.df[self.df["res_name"] == amnoacids.one_to_three[res_name]][self.df["res_num"] == res_num][
                self.df["atom_name"] == atom_name
            ].index.tolist()
            near_idx = self.kdtree.query_ball_point(self.kdtree.data[idx[0]], r)
        return [self.df.iloc[i] for i in near_idx]

    # @profile
    def all_atom_pairs_within_distance(self, r):
        if r == np.inf:
            pairs = [
                (self.df.iloc[i], self.df.iloc[j]) for i in range(len(self.df.index)) for j in range(len(self.df.index))
            ]
            return pairs
        else:
            near_idx = self.kdtree.query_pairs(r)
            pairs = [(self.df.iloc[self.kdtree_idx[i]], self.df.iloc[self.kdtree_idx[j]]) for i, j in near_idx]
            return pairs

    @property
    # @profile
    def sequence(self):
        seqs = dict()
        for chain_id, sel in self.df.groupby("chainID"):
            seq_df = sel.groupby("res_num")[["res_num", "res_name"]].first()
            i_0, i_n = seq_df["res_num"].min(), seq_df["res_num"].max()
            seq_dict = {k: aminoacids.three_to_one[v] for k, v in seq_df[["res_num", "res_name"]].values}
            seqs[chain_id] = ([seq_dict.get(i, None) for i in range(i_0, i_n + 1)], i_0)
        return seqs

    # @profile
    def regularize(self):
        print("here")
        atoms = []
        res_idx = 1
        for chain_id, sel in self.df.groupby("chainID"):
            seq_df = sel.groupby("res_num")[["res_num", "res_name"]].first()
            print(seq_df)
            i_0, i_n = seq_df["res_num"].min(), seq_df["res_num"].max()
            seq_dict = {k: aminoacids.three_to_one[v] for k, v in seq_df[["res_num", "res_name"]].values}
            seq = [aminoacids.one_to_three.get(seq_dict.get(i, "GLY"), "G") for i in range(i_0, i_n + 1)]
            print("".join(seq))

            df_values = sel.values
            self.df[sel.df.chainID == chain_id, "res_num"] = df_values[:, 6] - df_values[:, 6].min() + 1

            df_values = sel.values
            pres_list = list(zip(df_values[:, 2], df_values[:, 4], df_values[:, 6]))
            pres_set = set(pres_list)
            if len(pres_list) != len(pres_set):
                raise ValueError

            for i, aa in enumerate(seq):
                for atom_name in aminoacids.atom_names.get(aa, []):
                    if (atom_name, aa, i + 1) not in pres_set:
                        atoms.append(
                            [
                                "ATOM",  # record_name
                                0,  # serial
                                atom_name,  # atom_name
                                np.nan,  # altLoc
                                aa,  # res_name
                                np.nan,  # chainID
                                i + res_idx,  # res_num
                                np.nan,  # iCode
                                np.nan,  # x
                                np.nan,  # y
                                np.nan,  # z
                                np.nan,  # occupancy
                                np.nan,  # tempFactor
                                atom_name[0],  # element
                                np.nan,  # charge
                                np.nan,  # chemical shift
                            ]
                        )
        res_idx += i + 1

        self.df = pd.concat((self.df, pd.DataFrame(atoms, columns=self.df.columns.values)))

        self.df = self.df[
            [atm in aminoacids.atom_names[res] for res, atm in zip(self.df.values[:, 4], self.df.values[:, 2])]
        ]

        self.df = self.df.sort_values(["res_num", "atom_name"])
        self.df["serial"] = list(range(len(self.df.index)))
        self.df = self.df.reset_index(drop=True)

        self.build_kdtree()
        return self

    # @profile
    def atom_pair_info(self, i, j):
        if (i, j) not in self._atom_pair_info.keys():
            r1 = self.df.iloc[i]
            r2 = self.df.iloc[j]
            d = np.sqrt((r1.x - r2.x) ** 2 + (r1.y - r2.y) ** 2 + (r1.z - r2.z) ** 2)
            bt = aminoacids.bond_types[(r1.res_name, r1.atom_name, r2.atom_name)]
            self._atom_pair_info[(i, j)] = (r1.element, r2.element, bt, d)
            self._atom_pair_info[(j, i)] = (r2.element, r1.element, bt, d)
        return self._atom_pair_info[(i, j)]

    @property
    # @profile
    def bonds(self):
        if self._bonds is None:
            self._bonds = []
            for _, res in self.df.groupby("res_num"):
                for r2, r1 in combinations(res.itertuples(), 2):
                    if (
                        (r1.atom_name in ["H1", "H2", "H3"] and r2.atom_name == "N")
                        or (r2.atom_name in ["H1", "H2", "H3"] and r1.atom_name == "N")
                        or (r1.atom_name == "OXT" and r2.atom_name == "C")
                        or (r2.atom_name == "OXT" and r1.atom_name == "C")
                    ):
                        self._bonds.append(tuple(sorted((r1.Index, r2.Index))))
                    elif (r1.res_num == r2.res_num) and (
                        r2.atom_name in aminoacids.bonds.get(r1.res_name, dict()).get(r1.atom_name, [])
                    ):
                        self._bonds.append(tuple(sorted((r1.Index, r2.Index))))

            sg = self.df[self.df.atom_name == "SG"]
            for _sg_0 in sg.itertuples():
                for _sg_1 in sg.itertuples():
                    if _sg_0 != _sg_1:
                        d = np.sqrt((_sg_0.x - _sg_1.x) ** 2 + (_sg_0.y - _sg_1.y) ** 2 + (_sg_0.z - _sg_1.z) ** 2)
                        if d < 2.1:
                            self._bonds.append(tuple(sorted((_sg_0.Index, _sg_1.Index))))

            c = self.df[self.df.atom_name == "C"]
            for _c in c.itertuples():
                n = self.df[(self.df.res_num == _c.res_num + 1) & (self.df.atom_name == "N")]
                for _n in n.itertuples():
                    self._bonds.append((_c.Index, _n.Index))

            self._bonds = set(self._bonds) | {e[::-1] for e in self._bonds}
        return self._bonds

    @property
    # @profile
    def dmat(self):
        if self._dmat is None:
            xyz = self.df[["x", "y", "z"]].values
            self._dmat = np.sqrt(np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, 2))
        return self._dmat


def pdb_to_dataframes(f):
    models = [[]]
    for ln in f.readlines():
        if ln[:6] == "MODEL " and len(models[-1]) > 0:
            models.append([])
        elif ln[:6] == "ATOM  ":
            models[-1].append(ln)
    model_dfs = []
    for mdl_lns in models:
        df = pd.read_fwf(
            StringIO("".join(mdl_lns)),
            colspecs=[
                (0, 6),
                (6, 11),
                (11, 16),
                (16, 17),
                (17, 20),
                (20, 22),
                (22, 26),
                (26, 27),
                (27, 38),
                (38, 46),
                (46, 54),
                (54, 60),
                (60, 66),
                (77, 78),
                (78, 80),
            ],
            names=[
                "record_name",
                "serial",
                "atom_name",
                "altLoc",
                "res_name",
                "chainID",
                "res_num",
                "iCode",
                "x",
                "y",
                "z",
                "occupancy",
                "tempFactor",
                "element",
                "charge",
            ],
        )

        model_dfs.append(df)
    return model_dfs


class PDB(object):
    @classmethod
    # @profile
    def from_database(cls, pdb_id):
        file_name = f"{pdb_id}.pdb"
        if os.path.isfile(file_name):
            with open(file_name, "r") as f:
                pdb_str = f.read()
        else:
            fullurl = "http://www.rcsb.org/pdb/download/downloadFile.do?fileFormat=pdb&compression=NO"
            fullurl += "&structureId=" + pdb_id
            req = Request(fullurl)
            f = urlopen(req, timeout=10)
            pdb_str = f.read()
            pdb_str = pdb_str.decode("unicode_escape")
            with open(file_name, "w") as f:
                f.write(pdb_str)
        return cls(pdb_str)

    @classmethod
    # @profile
    def from_file(cls, path):
        try:
            return cls(path.read())
        except:
            with open(path, "r") as f:
                p = cls(f.read())
            return p

    @classmethod
    # @profile
    def from_dataframe(cls, df):
        pdb = cls()
        pdb.models = [PDBModel(df)]
        return pdb

    @classmethod
    def from_sequence(cls, seq):
        seq = [aminoacids.one_to_three[e] for e in list(seq)]
        fixer = PDBFixer(pdbid="2qmt")
        fixer.removeChains([1])

        chain = next(fixer.topology.chains())
        orig_len = len(list(chain.residues()))

        chain = next(fixer.topology.chains())
        index = list(chain.residues())[-1].index

        fixer.missingResidues = {(0, index + 1): seq}

        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        to_delete = []
        for res in list(next(fixer.topology.chains()).residues())[:orig_len]:
            for atom in res.atoms():
                to_delete.append(atom)
        modeller = app.Modeller(fixer.topology, fixer.positions)
        modeller.delete(to_delete)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

        with StringIO() as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
            p = PDB(f.getvalue())
        return p

    # @profile
    def __init__(self, pdb_str="", *args, **kwargs):
        if len(pdb_str) > 0:
            self.models = [PDBModel(df, *args, **kwargs) for df in pdb_to_dataframes(StringIO(pdb_str))]

    # @profile
    def to_file(self, f):
        try: 
            f.seek(0)
            for i, mdl in enumerate(self.models):
                f.write("MODEL{:9d}".format(i + 1) + " " * 66 + "\n")
                f.write(self._model_to_string(mdl))
                f.write("ENDMDL" + " " * 74 + "\n")
        except AttributeError:
            with open(f, 'w') as _f:
                for i, mdl in enumerate(self.models):
                    _f.write("MODEL{:9d}".format(i + 1) + " " * 66 + "\n")
                    _f.write(self._model_to_string(mdl))
                    _f.write("ENDMDL" + " " * 74 + "\n")

    # @profile
    def _model_to_string(self, mdl):
        s = ""
        for j, row in mdl.df.iterrows():
            s += "{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                str(row["record_name"]),
                row["serial"],
                (f" {row['atom_name']}" if len(row["atom_name"]) < 4 else str(row["atom_name"])),
                (" " if row["altLoc"] != row["altLoc"] else str(row["altLoc"])),
                str(row["res_name"]),
                (" " if row["chainID"] != row["chainID"] else str(row["chainID"])),
                int(row["res_num"]),
                (" " if row["iCode"] != row["iCode"] else str(row["iCode"])),
                row["x"],
                row["y"],
                row["z"],
                row["occupancy"],
                row["tempFactor"],
                str(row["element"]),
                (" " if row["charge"] != row["charge"] else str(row["charge"])),
            )
        return s

    def combine_chains(self):
        for i in range(len(self.models)):
            res_idx_max = 0
            for chain_id, sel in list(self.models[i].df.groupby("chainID")):
                self.models[i].df.loc[self.models[i].df.chainID == chain_id, "res_num"] = (
                    sel.res_num - sel.res_num.min() + res_idx_max + 1
                )
                res_idx_max += sel.res_num.max() - sel.res_num.min()
                self.models[i].df.loc[self.models[i].df.chainID == chain_id, "chainID"] = "A"

    @property
    # @profile
    def sequence(self, model=0):
        return self.models[model].sequence

    def _dihed(self, u1, u2, u3, u4):
        a1 = u2 - u1
        a2 = u3 - u2
        a3 = u4 - u3

        v1 = np.cross(a1, a2)
        v1 = v1 / (v1 * v1).sum(-1)[:, None] ** 0.5
        v2 = np.cross(a2, a3)
        v2 = v2 / (v2 * v2).sum(-1)[:, None] ** 0.5
        porm = np.sign((v1 * a3).sum(-1))
        rad = np.arccos((v1 * v2).sum(-1) / ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5)
        rad[porm != 0] = rad[porm != 0] * porm[porm != 0]

        return rad

    def backbone_dihedrals(self):
        d_m = []
        for m in self.models:
            n_min = m.df.res_num.min()
            n_max = m.df.res_num.max()

            assert len(m.df[m.df.atom_name == "N"]) == n_max - n_min + 1
            assert len(m.df[m.df.atom_name == "CA"]) == n_max - n_min + 1
            assert len(m.df[m.df.atom_name == "C"]) == n_max - n_min + 1

            d = np.zeros((n_max - n_min + 1, 3))
            u_1 = m.df[(m.df.atom_name == "C") & (m.df.res_num < n_max)][["x", "y", "z"]].to_numpy()
            u_2 = m.df[(m.df.atom_name == "N") & (m.df.res_num > n_min)][["x", "y", "z"]].to_numpy()
            u_3 = m.df[(m.df.atom_name == "CA") & (m.df.res_num > n_min)][["x", "y", "z"]].to_numpy()
            u_4 = m.df[(m.df.atom_name == "C") & (m.df.res_num > n_min)][["x", "y", "z"]].to_numpy()
            d[1:, 0] = self._dihed(u_1, u_2, u_3, u_4)

            u_1 = m.df[(m.df.atom_name == "N") & (m.df.res_num < n_max)][["x", "y", "z"]].to_numpy()
            u_2 = m.df[(m.df.atom_name == "CA") & (m.df.res_num < n_max)][["x", "y", "z"]].to_numpy()
            u_3 = m.df[(m.df.atom_name == "C") & (m.df.res_num < n_max)][["x", "y", "z"]].to_numpy()
            u_4 = m.df[(m.df.atom_name == "N") & (m.df.res_num > n_min)][["x", "y", "z"]].to_numpy()
            d[:-1, 1] = self._dihed(u_1, u_2, u_3, u_4)

            u_1 = m.df[(m.df.atom_name == "CA") & (m.df.res_num < n_max)][["x", "y", "z"]].to_numpy()
            u_2 = m.df[(m.df.atom_name == "C") & (m.df.res_num < n_max)][["x", "y", "z"]].to_numpy()
            u_3 = m.df[(m.df.atom_name == "N") & (m.df.res_num > n_min)][["x", "y", "z"]].to_numpy()
            u_4 = m.df[(m.df.atom_name == "CA") & (m.df.res_num > n_min)][["x", "y", "z"]].to_numpy()
            d[:-1, 2] = self._dihed(u_1, u_2, u_3, u_4)
            d_m.append(d)
        return d_m

    # @profile
    def regularize(self, pH=7.0, replace_nonstandard_residues=False, mutations=[], d_cutoff=5, relax=False):
        out_s = ""
        for i_mdl, mdl in enumerate(self.models[:]):
            mdl.df = mdl.df.sort_values(["chainID", "res_num"])
            mdl.df.loc[:, "serial"] = np.arange(1, len(mdl.df.index) + 1)
            with StringIO() as f:
                f.write(self._model_to_string(mdl))
                f.seek(0)
                fixer = PDBFixer(pdbfile=f)

            modeller = app.Modeller(fixer.topology, fixer.positions)
            fixer.topology = modeller.topology
            fixer.positions = modeller.positions
            n_chains = len(list(fixer.topology.chains()))
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            if replace_nonstandard_residues:
                fixer.replaceNonstandardResidues()
            fixer.applyMutations(mutations, list(fixer.topology.chains())[0].id)
            fixer.removeHeterogens(True)
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(pH)

            free_atoms = set()

            if len(mutations) > 0:
                mut_resid = []
                for e in mutations:
                    mut_resid.append(int(e.split("-")[1]))

                mut_atoms = []
                for atom in fixer.topology.atoms():
                    if int(atom.residue.id) in mut_resid:
                        mut_atoms.append(atom.index)
                mut_atom_pos = np.array([fixer.positions[a_idx].value_in_unit(unit.angstroms) for a_idx in mut_atoms])
                pos = np.array(fixer.positions.value_in_unit(unit.angstroms))
                d_to_closest = np.min(np.sqrt(np.sum((pos[:, None, :] - mut_atom_pos[None, :, :]) ** 2, 2)), 1)
                near_mut_atoms = np.where(d_to_closest < d_cutoff)[0]

                for a_idx in mut_atoms:
                    free_atoms.add(a_idx)
                for a_idx in near_mut_atoms:
                    free_atoms.add(a_idx)

            fixed_atoms = set()
            for atom in fixer.topology.atoms():
                if atom.element.name != "hydrogen":
                    fixed_atoms.add(atom.index)

            fixed_atoms = fixed_atoms - free_atoms

            out_s += "MODEL{:9d}".format(i_mdl + 1) + " " * 66 + "\n"
            with StringIO() as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
                if relax:
                    f.seek(0)
                    p = PDB.from_file(f).relax()
                    f.seek(0)
                    f.truncate(0)
                    PDBFile.writeFile(p.topology, p.positions, f)
                f.seek(0)
                out_s += "".join(list(f.readlines())[1:-1])
            out_s += "ENDMDL" + " " * 74 + "\n"

        p = PDB(out_s)
        for i in range(len(p.models)):
            p.models[i].df = p.models[i].df.reset_index(drop=True)
        return p  # , sorted(list(fixed_atoms))

    def relax(self, fixed_atoms=[]):

        with StringIO() as f:
            self.to_file(f)
            f.seek(0)
            p = PDBFile(f)

        forcefield = app.ForceField("amber99sb.xml", "amber99_obc.xml")

        # Creating System
        system = forcefield.createSystem(p.topology, constraints=app.forcefield.HBonds)

        force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", 0.0 * unit.kilocalories_per_mole / unit.angstroms ** 2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        for i, atom_index in enumerate(fixed_atoms):
            force.addParticle(atom_index, p.positions[atom_index])
        system.addForce(force)

        integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
        # Creating simulation context
        simulation = app.Simulation(p.topology, system, integrator)
        simulation.context.setPositions(p.positions)
        # Minimizing System
        simulation.minimizeEnergy(maxIterations=100)
        # Adding Reporters
        simulation.reporters.append(app.PDBReporter("output_exercise1.pdb", 5))
        simulation.reporters.append(
            app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True)
        )
        # Running simulation
        for i in range(1000):
            alpha = max(min((i - 900) ** 2 / (900 ** 2), 1), 0)
            integrator.setTemperature((alpha * 290 + 10) * unit.kelvin)
            simulation.context.setParameter("k", (1 - alpha) * 10.0 * unit.kilocalories_per_mole / unit.angstroms ** 2)

            simulation.step(10)

        p.positions = simulation.context.getState(getPositions=True).getPositions()

        with StringIO() as f:
            PDBFile.writeFile(p.topology, p.positions, f)
            p = PDB(f.getvalue())
        return p


# @profile
def main():
    p = PDB.from_file("./2LGI.pdb")
    m = p.models[0]
    pairs = m.all_atom_pairs_within_distance(3)
    for atm1, atm2 in pairs:
        print("=" * 64)
        print(aminoacids.three_to_one[atm1.res_name], atm1.res_num, atm1.atom_name)
        print(aminoacids.three_to_one[atm2.res_name], atm2.res_num, atm2.atom_name)
        print("=" * 64)


if __name__ == "__main__":
    main()
