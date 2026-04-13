import pytest
from ase import Atoms
from assyst.crystals import sample_space_groups, Formulas, pyxtal
from assyst.perturbations import apply_perturbations, Rattle, Stretch, Series, rattle
from assyst.relax import relax, Relax
from assyst.calculators import Morse

def test_full_workflow_lineage():
    # 1. Generate
    # Using pyxtal directly for more reliability in test
    s1 = pyxtal(1, species=["Cu"], num_ions=[2])
    assert isinstance(s1, Atoms)

    uuid1 = s1.info.get("uuid")
    assert uuid1 is not None
    assert s1.info.get("seed") == uuid1
    assert "lineage" not in s1.info or len(s1.info["lineage"]) == 0

    # 2. Perturb
    perturbations = [Series((Rattle(0.1), Stretch(0.1, 0.1)))]
    perturbed = list(apply_perturbations([s1], perturbations))
    assert len(perturbed) == 1
    s2 = perturbed[0]

    uuid2 = s2.info.get("uuid")
    assert uuid2 is not None
    assert uuid2 != uuid1
    assert s2.info.get("seed") == uuid1
    # Lineage should contain uuid1 and the uuid after Rattle
    assert s2.info["lineage"][0] == uuid1
    assert len(s2.info["lineage"]) == 2
    uuid_after_rattle = s2.info["lineage"][1]

    # 3. Relax
    calc = Morse()
    relaxed = list(relax(Relax(max_steps=2), calc, [s2]))
    assert len(relaxed) == 1
    s3 = relaxed[0]

    uuid3 = s3.info.get("uuid")
    assert uuid3 is not None
    assert uuid3 != uuid2
    assert s3.info.get("seed") == uuid1
    assert s3.info["lineage"] == [uuid1, uuid_after_rattle, uuid2]

def test_individual_perturbations():
    s = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)
    s.info["uuid"] = "initial-uuid"

    # Rattle
    r = Rattle(0.1)
    s_perturbed = r(s.copy())
    assert s_perturbed.info["uuid"] != "initial-uuid"
    assert s_perturbed.info["lineage"] == ["initial-uuid"]

    # Stretch
    st = Stretch(0.1, 0.1)
    s_stretched = st(s.copy())
    assert s_stretched.info["uuid"] != "initial-uuid"
    assert s_stretched.info["lineage"] == ["initial-uuid"]

def test_series_perturbation():
    s = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)
    s.info["uuid"] = "uuid0"

    ser = Series((Rattle(0.1), Stretch(0.1, 0.1)))
    s_final = ser(s.copy())

    # Should have two new UUIDs in lineage
    assert len(s_final.info["lineage"]) == 2
    assert s_final.info["lineage"][0] == "uuid0"
    assert s_final.info["uuid"] != s_final.info["lineage"][1]

def test_inplace_function_no_uuid_change():
    s = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)
    s.info["uuid"] = "fixed-uuid"

    # Calling rattle (the function) directly
    rattle(s, 0.1)
    assert s.info["uuid"] == "fixed-uuid"
    assert "lineage" not in s.info

def test_apply_perturbations_with_function():
    s = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)
    s.info["uuid"] = "orig"

    # Using the raw rattle function in apply_perturbations
    perturbed = list(apply_perturbations([s], [lambda atoms: rattle(atoms, 0.1)]))
    assert len(perturbed) == 1
    assert perturbed[0].info["uuid"] != "orig"
    assert perturbed[0].info["lineage"] == ["orig"]

def test_relax_lineage():
    s = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)
    s.info["uuid"] = "before-relax"
    s.calc = Morse().get_calculator()

    rel = Relax(max_steps=1)
    s_relaxed = rel.relax(s)

    assert s_relaxed.info["uuid"] != "before-relax"
    assert s_relaxed.info["lineage"] == ["before-relax"]

def test_no_initial_uuid():
    # If no initial UUID, lineage should be empty
    s = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)

    r = Rattle(0.1)
    s_perturbed = r(s)
    assert s_perturbed.info["uuid"] is not None
    assert s_perturbed.info["seed"] == s_perturbed.info["uuid"]
    assert "lineage" not in s_perturbed.info

def test_lineage_not_shared_with_parent():
    s1 = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)
    s1.info["uuid"] = "uuid1"

    # First modification
    r = Rattle(0.1)
    s2 = r(s1.copy())
    assert s2.info["uuid"] != "uuid1"
    assert s2.info["lineage"] == ["uuid1"]

    # Original should NOT have uuid1 in lineage
    assert "lineage" not in s1.info or s1.info["lineage"] == []

    # Second modification
    s3 = r(s2.copy())
    assert s3.info["lineage"] == ["uuid1", s2.info["uuid"]]

    # s2 lineage should NOT be affected by s3 modification
    assert s2.info["lineage"] == ["uuid1"]

def test_all_inplace_functions_via_apply_perturbations():
    from assyst.perturbations import rattle, stretch, element_scaled_rattle
    s = Atoms("Cu2", positions=[[0,0,0], [1,1,1]], cell=[3,3,3], pbc=True)
    s.info["uuid"] = "orig"

    # rattle
    perturbed = list(apply_perturbations([s], [lambda atoms: rattle(atoms, 0.1)]))
    assert len(perturbed) == 1
    assert perturbed[0].info["uuid"] != "orig"
    assert perturbed[0].info["lineage"] == ["orig"]

    # stretch
    perturbed = list(apply_perturbations([s], [lambda atoms: stretch(atoms, 0.1, 0.1)]))
    assert len(perturbed) == 1
    assert perturbed[0].info["uuid"] != "orig"
    assert perturbed[0].info["lineage"] == ["orig"]

    # element_scaled_rattle
    ref = {"Cu": 2.5}
    perturbed = list(apply_perturbations([s], [lambda atoms: element_scaled_rattle(atoms, 0.1, ref)]))
    assert len(perturbed) == 1
    assert perturbed[0].info["uuid"] != "orig"
    assert perturbed[0].info["lineage"] == ["orig"]
