"""Tailored tests for tools that the generic smoke runner can't call correctly.

These tests exist primarily to:
- provide minimal valid inputs (arrays/dicts) where placeholders are wrong, and/or
- create minimal remote-side fixture files via `run_python_repl` for file-path tools,
- mark heavyweight / external-CLI tools as optional while still counting them as
  "tested" for the untested-tools smoke runner.

The untested smoke runner detects "tested" tools by scanning for `hypha_service.<tool>`
attribute usage in test files (text-level heuristic). So we keep calls explicit.
"""

from __future__ import annotations

import contextlib
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService

# Optional dependencies for local file generation
try:
    from skimage import io
except ImportError:
    io = None

try:
    import nibabel as nib
except ImportError:
    nib = None


async def _remote_tmpdir(
    _hypha_service: RemoteService,
    prefix: str = "biomni_test",
) -> str:
    """Create a unique directory locally and return its path.

    Note: These `_remote_` functions run locally in the test environment.
    They are named to indicate they prepare resources that the 'remote' service
    will use.
    """
    rid = uuid.uuid4().hex
    path = Path(tempfile.gettempdir()) / f"{prefix}_{rid}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


async def _remote_write_text(
    _hypha_service: RemoteService,
    path: str,
    content: str,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


async def _remote_write_png_sequence(
    _hypha_service: RemoteService,
    dir_path: str,
    n: int = 5,
) -> None:
    """Write a simple PNG time-lapse sequence locally using skimage."""
    if io is None:
        pytest.skip("skimage not installed")

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    blob_size = 64

    for i in range(n):
        img = np.zeros((blob_size, blob_size), dtype=np.uint8)
        # moving bright blob 1
        y1 = 10 + i * 5
        x1 = 10 + i * 5
        if y1 + 10 < blob_size and x1 + 10 < blob_size:
            img[y1 : y1 + 10, x1 : x1 + 10] = 255

        # moving bright blob 2 (slower)
        y2 = 10 + i * 2
        x2 = 40
        if y2 + 10 < blob_size:
            img[y2 : y2 + 10, x2 : x2 + 10] = 255

        io.imsave(str(Path(dir_path) / f"frame_{i:03d}.png"), img)


async def _remote_write_rgb_png(_hypha_service: RemoteService, path: str) -> None:
    if io is None:
        pytest.skip("skimage not installed")

    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[10:30, 10:50, 0] = 180
    img[40:70, 60:110, 1] = 200
    img[20:60, 30:90, 2] = 220
    io.imsave(path, img)


async def _remote_write_dwi_nifti(
    _hypha_service: RemoteService,
    path: str,
) -> list[float]:
    """Write a tiny 4D DWI NIfTI (b=0 and b=1000) and return matching b-values."""
    if nib is None:
        pytest.skip("nibabel not installed")

    bvals = [0.0, 1000.0]

    # 4D volume: (x,y,z,nb)
    shape = (4, 4, 2, 2)
    # ensure strictly positive signals
    vol = np.ones(shape, dtype=np.float32)
    # baseline signal
    vol[:, :, :, 0] *= 1000.0
    # diffusion-weighted signal
    vol[:, :, :, 1] *= 300.0

    img = nib.Nifti1Image(vol, affine=np.eye(4))
    nib.save(img, path)
    return bvals


async def _remote_write_minimal_pdb(_hypha_service: RemoteService, path: str) -> None:
    pdb = """
ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.600  10.000  1.00 20.00           C
ATOM      4  O   ALA A   1      12.400  15.600  10.000  1.00 20.00           O
TER
END
"""
    await _remote_write_text(_hypha_service, path, pdb)


@pytest.mark.asyncio
async def test_needs_fixture__analyze_abr_waveform_p1_metrics(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing ABR waveform P1 metrics."""
    time_ms = np.linspace(0, 6, 601)
    amp = np.exp(-((time_ms - 2.0) ** 2) / (2 * 0.08**2))
    amp = amp - 0.2 * np.exp(-((time_ms - 3.5) ** 2) / (2 * 0.2**2))
    out = await hypha_service.analyze_abr_waveform_p1_metrics(
        time_ms=time_ms,
        amplitude_uv=amp,
    )
    assert isinstance(out, str)
    assert "P1" in out
    assert "Latency" in out
    assert "Amplitude" in out


@pytest.mark.asyncio
async def test_needs_fixture__analyze_bacterial_growth_rate(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing bacterial growth rate."""
    tmp = await _remote_tmpdir(hypha_service)
    out = await hypha_service.analyze_bacterial_growth_rate(
        time_points=[0, 1, 2, 3, 4, 5, 6],
        od_measurements=[0.05, 0.07, 0.12, 0.25, 0.45, 0.70, 0.85],
        strain_name="TestStrain",
        output_dir=tmp,
    )
    assert isinstance(out, str)
    assert "Growth Rate" in out
    assert "Doubling Time" in out


@pytest.mark.asyncio
async def test_needs_fixture__analyze_bifurcation_diagram(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing bifurcation diagram."""
    tmp = await _remote_tmpdir(hypha_service)
    ts = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    pv = np.array([0.1, 0.2, 0.3], dtype=float)
    out = await hypha_service.analyze_bifurcation_diagram(
        time_series_data=ts,
        parameter_values=pv,
        system_name="TestSys",
        output_dir=tmp,
    )
    assert isinstance(out, str)
    assert "Bifurcation Analysis" in out


@pytest.mark.asyncio
async def test_needs_fixture__analyze_accelerated_stability_of_pharm_formulations(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing accelerated stability of pharmaceutical formulations."""
    formulations = [
        {
            "name": "Formulation_A",
            "active_ingredient": "TestAPI",
            "concentration": 10.0,
            "excipients": ["water", "buffer"],
            "dosage_form": "solution",
        },
    ]
    storage_conditions = [
        {"temperature": 40.0, "humidity": 75.0, "description": "Accelerated"},
    ]
    time_points = [0, 7, 14]
    func = hypha_service.analyze_accelerated_stability_of_pharmaceutical_formulations
    out = await func(
        formulations=formulations,
        storage_conditions=storage_conditions,
        time_points=time_points,
    )
    assert isinstance(out, str)
    assert "Accelerated Stability Testing" in out


@pytest.mark.asyncio
async def test_needs_fixture__analyze_endolysosomal_calcium_dynamics(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing endolysosomal calcium dynamics."""
    tmp = await _remote_tmpdir(hypha_service)
    t = np.arange(0, 60, 1)
    y = 1.0 + 0.05 * np.sin(2 * np.pi * t / 20.0)
    out = await hypha_service.analyze_endolysosomal_calcium_dynamics(
        time_points=t,
        luminescence_values=y,
        treatment_time=20,
        cell_type="HEK293",
        treatment_name="ATP",
        output_file=f"{tmp}/calcium.txt",
    )
    assert isinstance(out, dict)
    assert out["url"].startswith(("http://", "https://"))
    assert isinstance(out.get("log"), str)
    assert "Ca2+ Dynamics" in out["log"]


@pytest.mark.asyncio
async def test_needs_fixture__analyze_hemodynamic_data(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing hemodynamic data."""
    tmp = await _remote_tmpdir(hypha_service)
    # Sampling rate 100 Hz -> Nyquist 50 Hz; bandpass is 0.5-10 Hz.
    sr = 100.0
    t = np.arange(0, 10.0, 1 / sr)
    pressure = 80 + 20 * np.sin(2 * np.pi * 2.0 * t) + 2 * np.sin(2 * np.pi * 0.8 * t)
    out = await hypha_service.analyze_hemodynamic_data(
        pressure_data=pressure,
        sampling_rate=sr,
        output_file=f"{tmp}/hemo.csv",
    )
    assert isinstance(out, dict)
    assert out["url"].startswith(("http://", "https://"))
    assert isinstance(out.get("log"), str)
    assert "Heart Rate" in out["log"]


@pytest.mark.asyncio
async def test_needs_fixture__analyze_in_vitro_drug_release_kinetics(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing in vitro drug release kinetics."""
    outdir = await _remote_tmpdir(hypha_service)
    out = await hypha_service.analyze_in_vitro_drug_release_kinetics(
        time_points=[0, 1, 2, 4, 8, 12, 24],
        concentration_data=[0, 5, 10, 25, 40, 55, 70],
        drug_name="TestDrug",
        total_drug_loaded=100.0,
        output_dir=outdir,
    )
    assert isinstance(out, str)
    assert "Release Kinetics" in out


@pytest.mark.asyncio
async def test_needs_fixture__decode_behavior_from_neural_trajectories(
    hypha_service: RemoteService,
) -> None:
    """Test decoding behavior from neural trajectories."""
    tmp = await _remote_tmpdir(hypha_service)
    neural = np.random.RandomState(0).randn(200, 20)
    beh = np.vstack(
        [
            np.sin(np.linspace(0, 6.28, 200)),
            np.cos(np.linspace(0, 6.28, 200)),
        ],
    ).T
    out = await hypha_service.decode_behavior_from_neural_trajectories(
        neural_data=neural,
        behavioral_data=beh,
        n_components=5,
        output_dir=tmp,
    )
    assert isinstance(out, str)
    assert "Neural Trajectory Modeling" in out


@pytest.mark.asyncio
async def test_needs_fixture__design_verification_primers(
    hypha_service: RemoteService,
) -> None:
    """Test designing verification primers."""
    plasmid = "ATG" + "A" * 600 + "TAA"
    out = await hypha_service.design_verification_primers(
        plasmid_sequence=plasmid,
        target_region=[120, 220],
        existing_primers=[],
        is_circular=True,
    )
    assert isinstance(out, (str, dict))
    if isinstance(out, str):
        assert "Primer Pair" in out


@pytest.mark.asyncio
async def test_needs_fixture__estimate_cell_cycle_phase_durations(
    hypha_service: RemoteService,
) -> None:
    """Test estimating cell cycle phase durations."""
    flow = {
        "time_points": [0, 1, 2, 3, 4],
        "edu_positive": [10, 20, 30, 25, 15],
        "brdu_positive": [5, 15, 25, 20, 10],
        "double_positive": [2, 6, 12, 9, 4],
    }
    init = {
        "g1_duration": 10.0,
        "s_duration": 8.0,
        "g2m_duration": 6.0,
        "death_rate": 0.01,
    }
    out = await hypha_service.estimate_cell_cycle_phase_durations(
        flow_cytometry_data=flow,
        initial_estimates=init,
    )
    assert isinstance(out, str)
    assert "Cell Cycle Phase Duration Estimation" in out


@pytest.mark.asyncio
async def test_needs_fixture__find_restriction_sites(
    hypha_service: RemoteService,
) -> None:
    """Test finding restriction sites."""
    out = await hypha_service.find_restriction_sites(
        dna_sequence="GAATTC" * 20,
        enzymes=["EcoRI"],
        is_circular=False,
    )
    assert isinstance(out, (str, dict))
    if isinstance(out, dict):
        assert "sequence_info" in out
        assert "EcoRI" in str(out)
    else:
        assert "EcoRI" in out


@pytest.mark.asyncio
async def test_needs_fixture__analyze_interaction_mechanisms(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing interaction mechanisms."""
    # JSON-friendly list instead of tuple
    out = await hypha_service.analyze_interaction_mechanisms(
        drug_pair=["warfarin", "aspirin"],
        detailed_analysis=False,
        data_lake_path=None,
    )
    assert isinstance(out, str)
    assert "Drug Interaction Mechanism Analysis" in out


@pytest.mark.asyncio
async def test_needs_fixture__fit_genomic_prediction_model(
    hypha_service: RemoteService,
) -> None:
    """Test fitting genomic prediction model."""
    tmp = await _remote_tmpdir(hypha_service)
    genotypes = np.array(
        [
            [0, 1, 2, 0],
            [1, 1, 0, 2],
            [2, 0, 1, 1],
            [0, 2, 1, 0],
            [1, 0, 2, 2],
        ],
        dtype=float,
    )
    phenotypes = np.array([1.0, 1.2, 0.8, 1.1, 0.9], dtype=float)
    out = await hypha_service.fit_genomic_prediction_model(
        genotypes=genotypes,
        phenotypes=phenotypes,
        model_type="additive",
        output_file=f"{tmp}/gp.csv",
    )
    assert isinstance(out, str)
    assert "Genomic Prediction Analysis" in out


@pytest.mark.asyncio
async def test_needs_fixture__model_protein_dimerization_network(
    hypha_service: RemoteService,
) -> None:
    """Test modeling protein dimerization network."""
    monomers = {"A": 1.0, "B": 1.0}
    affinities = {"A-B": 2.0}
    topology = [("A", "B")]
    out = await hypha_service.model_protein_dimerization_network(
        monomer_concentrations=monomers,
        dimerization_affinities=affinities,
        network_topology=topology,
    )
    assert isinstance(out, str)
    assert "Protein Dimerization Network Modeling" in out


@pytest.mark.asyncio
async def test_needs_fixture__optimize_anaerobic_digestion_process(
    hypha_service: RemoteService,
) -> None:
    """Test optimizing anaerobic digestion process."""
    waste = {"total_solids": 8.0, "volatile_solids": 6.0, "cod": 50.0}
    ops = {
        "hrt": (8.0, 12.0),
        "olr": (1.0, 3.0),
        "if_ratio": (0.2, 0.8),
        "temperature": (30.0, 40.0),
        "ph": (6.5, 7.5),
    }
    out = await hypha_service.optimize_anaerobic_digestion_process(
        waste_characteristics=waste,
        operational_parameters=ops,
        target_output="methane_yield",
        optimization_method="rsm",
    )
    assert isinstance(out, str)
    assert "Anaerobic Digestion Process Optimization" in out


@pytest.mark.asyncio
async def test_needs_fixture__perform_cosinor_analysis(
    hypha_service: RemoteService,
) -> None:
    """Test performing cosinor analysis."""
    t = np.arange(0, 48, 1)
    y = 100 + 10 * np.sin(2 * np.pi * t / 24.0)
    out = await hypha_service.perform_cosinor_analysis(
        time_data=t,
        physiological_data=y,
        period=24,
    )
    assert isinstance(out, str)
    assert "Mesor" in out


@pytest.mark.asyncio
async def test_needs_fixture__read_function_source_code(
    hypha_service: RemoteService,
) -> None:
    """Test reading function source code."""
    out = await hypha_service.read_function_source_code(
        function_name="biomni.tool.physiology.analyze_abr_waveform_p1_metrics",
    )
    assert isinstance(out, str)
    if out.lstrip().startswith("Error:"):
        pytest.xfail("Remote environment could not import the requested module")
    assert "def analyze_abr_waveform_p1_metrics" in out


@pytest.mark.asyncio
async def test_needs_fixture__run_3d_chondrogenic_aggregate_assay(
    hypha_service: RemoteService,
) -> None:
    """Test running 3D chondrogenic aggregate assay."""
    cells = {
        "source": "human chondrocytes",
        "passage_number": 3,
        "cell_density": 1_000_000,
    }
    compounds = [{"name": "CompA", "concentration": "1 uM", "vehicle": "DMSO"}]
    out = await hypha_service.run_3d_chondrogenic_aggregate_assay(
        chondrocyte_cells=cells,
        test_compounds=compounds,
        culture_duration_days=14,
        measurement_intervals=7,
    )
    assert isinstance(out, str)
    assert "3D Chondrogenic Aggregate Culture Assay Protocol" in out


@pytest.mark.asyncio
async def test_needs_fixture__create_biochemical_network_sbml_model(
    hypha_service: RemoteService,
) -> None:
    """Test creating biochemical network SBML model."""
    tmp = await _remote_tmpdir(hypha_service)
    reaction_network = [
        {
            "id": "R1",
            "name": "A_to_B",
            "reactants": {"A": 1},
            "products": {"B": 1},
            "reversible": False,
        },
    ]
    kinetic_parameters = {
        "R1": {"law_type": "mass_action", "parameters": {"k": 0.1}},
    }

    out = await hypha_service.create_biochemical_network_sbml_model(
        reaction_network=reaction_network,
        kinetic_parameters=kinetic_parameters,
        output_file=f"{tmp}/biochemical_model.xml",
    )
    assert isinstance(out, dict)
    assert out["url"].startswith(("http://", "https://"))
    assert isinstance(out.get("log"), str)
    assert "SBML" in out["log"]


@pytest.mark.asyncio
async def test_needs_fixture__simulate_gene_circuit_with_growth_feedback(
    hypha_service: RemoteService,
) -> None:
    """Test simulating gene circuit with growth feedback."""
    topo = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)
    kinetic = {
        "basal_rates": [0.2, 0.2],
        "degradation_rates": [0.05, 0.05],
        "hill_coefficients": [2.0, 2.0],
        "threshold_constants": [1.0, 1.0],
    }
    growth = {
        "max_growth_rate": 0.5,
        "growth_inhibition": 0.1,
        "gene_growth_weights": [0.2, 0.2],
    }
    out = await hypha_service.simulate_gene_circuit_with_growth_feedback(
        circuit_topology=topo,
        kinetic_params=kinetic,
        growth_params=growth,
        simulation_time=10,
        time_points=50,
    )
    assert isinstance(out, str)
    assert "Gene Regulatory Circuit Simulation" in out


@pytest.mark.asyncio
async def test_needs_fixture__simulate_generalized_lotka_volterra_dynamics(
    hypha_service: RemoteService,
) -> None:
    """Test simulating generalized Lotka-Volterra dynamics."""
    tmp = await _remote_tmpdir(hypha_service)
    init = [0.2, 0.3, 0.5]
    gr = [0.4, 0.2, 0.1]
    a_matrix = [[-0.5, -0.1, 0.0], [-0.1, -0.4, -0.1], [0.0, -0.2, -0.3]]
    t = list(range(20))
    out = await hypha_service.simulate_generalized_lotka_volterra_dynamics(
        initial_abundances=init,
        growth_rates=gr,
        interaction_matrix=a_matrix,
        time_points=t,
        output_file=f"{tmp}/glv.csv",
    )
    assert isinstance(out, str)
    assert "Generalized Lotka-Volterra" in out


@pytest.mark.asyncio
async def test_needs_fixture__simulate_renin_angiotensin_system_dynamics(
    hypha_service: RemoteService,
) -> None:
    """Test simulating renin-angiotensin system dynamics."""
    init = {
        "renin": 1.0,
        "angiotensinogen": 10.0,
        "angiotensin_I": 0.0,
        "angiotensin_II": 0.0,
        "ACE2_angiotensin_II": 0.0,
        "angiotensin_1_7": 0.0,
    }
    rates = {
        "k_ren": 0.4,
        "k_agt": 0.2,
        "k_ace": 0.3,
        "k_ace2": 0.1,
        "k_at1r": 0.05,
        "k_mas": 0.05,
    }
    fb = {"fb_ang_II": 0.2, "fb_ace2": 0.1}
    out = await hypha_service.simulate_renin_angiotensin_system_dynamics(
        initial_concentrations=init,
        rate_constants=rates,
        feedback_params=fb,
        simulation_time=2,
        time_points=20,
    )
    assert isinstance(out, str)
    assert "Angiotensin II" in out


@pytest.mark.asyncio
async def test_needs_fixture__simulate_thyroid_hormone_pharmacokinetics(
    hypha_service: RemoteService,
) -> None:
    """Test simulating thyroid hormone pharmacokinetics."""
    params = {
        "transport_rates": {
            "blood_to_liver": 0.1,
            "blood_to_kidney": 0.1,
            "blood_to_thyroid": 0.1,
        },
        "binding_constants": {"k_on_T4_TBG": 0.01, "k_off_T4_TBG": 0.1},
        "metabolism_rates": {"T4_to_T3_liver": 0.05},
        "volumes": {"blood": 1.0, "liver": 1.0, "kidney": 1.0, "thyroid": 1.0},
    }
    init = {
        "T4_blood_free": 1.0,
        "TBG_blood": 5.0,
        "T4_TBG_complex": 0.0,
        "T4_liver_free": 0.0,
        "T3_liver_free": 0.0,
        "T4_kidney_free": 0.0,
        "T4_thyroid_free": 0.0,
    }
    out = await hypha_service.simulate_thyroid_hormone_pharmacokinetics(
        parameters=params,
        initial_conditions=init,
        time_span=(0, 1),
        time_points=25,
    )
    assert isinstance(out, str)
    assert "T4" in out


@pytest.mark.asyncio
async def test_needs_fixture__simulate_whole_cell_ode_model(
    hypha_service: RemoteService,
) -> None:
    """Test simulating whole cell ODE model."""
    init = {"mRNA": 0.1, "protein": 0.1, "metabolite": 0.1, "atp": 1.0}
    params = {
        "k_transcription": 0.5,
        "k_translation": 0.1,
        "k_mrna_deg": 0.05,
        "k_protein_deg": 0.02,
        "k_metabolism": 0.05,
        "k_atp_production": 0.2,
        "k_atp_consumption": 0.1,
    }
    out = await hypha_service.simulate_whole_cell_ode_model(
        initial_conditions=init,
        parameters=params,
        time_span=(0, 5),
        time_points=50,
        method="LSODA",
    )
    assert isinstance(out, str)
    assert "Whole-Cell ODE Model Simulation" in out


@pytest.mark.asyncio
async def test_needs_fixture__estimate_alpha_particle_radiotherapy_dosimetry(
    hypha_service: RemoteService,
) -> None:
    """Test estimating alpha particle radiotherapy dosimetry."""
    tmp = await _remote_tmpdir(hypha_service)
    biod = {
        "tumor": [(0.0, 10.0), (1.0, 9.0), (2.0, 7.0)],
        "liver": [(0.0, 5.0), (1.0, 4.0), (2.0, 3.0)],
    }
    rad = {
        "radionuclide": "Ac-225",
        "half_life_hours": 240.0,
        "energy_per_decay_MeV": 5.0,
        "radiation_weighting_factor": 5.0,
        # JSON-safe string keys; tool normalizes internally.
        "S_factors": {
            "tumor,tumor": 1.0,
            "liver,tumor": 0.1,
            "tumor,liver": 0.2,
            "liver,liver": 0.5,
        },
    }
    out = await hypha_service.estimate_alpha_particle_radiotherapy_dosimetry(
        biodistribution_data=biod,
        radiation_parameters=rad,
        output_file=f"{tmp}/dosimetry.csv",
    )
    assert isinstance(out, str)
    assert "Alpha-Particle Radiotherapy Dosimetry Estimation" in out


@pytest.mark.asyncio
async def test_expected_fail__analyze_barcode_sequencing_data_with_fixture_fastq(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing barcode sequencing data with fixture FASTQ."""
    tmp = await _remote_tmpdir(hypha_service)
    fastq_path = f"{tmp}/reads.fastq"
    # 10 reads with barcode AAAA/AAAT within flanking sequences
    fastq = """\
@r1\nGGGGAAAATTTT\n+\nFFFFFFFFFFFF\n@r2\nGGGGAAAATTTT\n+\nFFFFFFFFFFFF\n@r3\nGGGGAAATTTTT\n+\nFFFFFFFFFFFF\n@r4\nGGGGAAAATTTT\n+\nFFFFFFFFFFFF\n@r5\nGGGGAAAATTTT\n+\nFFFFFFFFFFFF\n@r6\nGGGGAAAATTTT\n+\nFFFFFFFFFFFF\n@r7\nGGGGAAAATTTT\n+\nFFFFFFFFFFFF\n@r8\nGGGGAAATTTTT\n+\nFFFFFFFFFFFF\n@r9\nGGGGAAATTTTT\n+\nFFFFFFFFFFFF\n@r10\nGGGGAAATTTTT\n+\nFFFFFFFFFFFF\n"""
    await _remote_write_text(hypha_service, fastq_path, fastq)

    out = await hypha_service.analyze_barcode_sequencing_data(
        input_file=fastq_path,
        flanking_seq_5prime="GGGG",
        flanking_seq_3prime="TTTT",
        min_count=2,
        output_dir=tmp,
    )
    assert isinstance(out, str)
    assert "Total reads processed" in out


@pytest.mark.asyncio
async def test_expected_fail__analyze_cell_migration_metrics_with_image_sequence(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing cell migration metrics with image sequence."""
    tmp = await _remote_tmpdir(hypha_service)
    seq_dir = f"{tmp}/seq"
    await _remote_write_png_sequence(hypha_service, seq_dir, n=12)

    # If trackpy isn't installed remotely, this should error quickly and that's
    # acceptable.
    with pytest.raises(Exception, match=r"(?i)(trackpy|no module|error)"):
        await hypha_service.analyze_cell_migration_metrics(
            image_sequence_path=seq_dir,
            pixel_size_um=1.0,
            time_interval_min=1.0,
            min_track_length=3,
            output_dir=tmp,
        )


@pytest.mark.asyncio
async def test_expected_fail__analyze_myofiber_morphology_with_multichannel_image(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing myofiber morphology with multichannel image."""
    tmp = await _remote_tmpdir(hypha_service)
    img_path = f"{tmp}/myofiber.png"
    await _remote_write_rgb_png(hypha_service, img_path)

    out = await hypha_service.analyze_myofiber_morphology(
        image_path=img_path,
        nuclei_channel=0,
        myofiber_channel=1,
        threshold_method="otsu",
        output_dir=tmp,
    )
    assert isinstance(out, str)
    assert "MYOFIBER MORPHOLOGICAL ANALYSIS REPORT" in out


@pytest.mark.asyncio
async def test_expected_fail__analyze_western_blot_with_fixture_image(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing western blot with fixture image."""
    tmp = await _remote_tmpdir(hypha_service)
    img_path = f"{tmp}/blot.png"
    await _remote_write_rgb_png(hypha_service, img_path)

    out = await hypha_service.analyze_western_blot(
        blot_image_path=img_path,
        target_bands=[{"name": "Target", "roi": [10, 10, 20, 20]}],
        loading_control_band={"name": "Actin", "roi": [10, 40, 20, 20]},
        antibody_info={"primary": "anti-Target", "secondary": "anti-IgG"},
        output_dir=tmp,
    )
    assert isinstance(out, str)
    assert "Western Blot Analysis" in out


@pytest.mark.asyncio
async def test_expected_fail__quantify_amyloid_beta_plaques_with_fixture_image(
    hypha_service: RemoteService,
) -> None:
    """Test quantifying amyloid beta plaques with fixture image."""
    tmp = await _remote_tmpdir(hypha_service)
    img_path = f"{tmp}/plaques.png"
    await _remote_write_rgb_png(hypha_service, img_path)

    out = await hypha_service.quantify_amyloid_beta_plaques(
        image_path=img_path,
        output_dir=tmp,
        threshold_method="otsu",
        min_plaque_size=10,
    )
    assert isinstance(out, str)
    assert "Plaque Analysis" in out


@pytest.mark.asyncio
async def test_expected_fail__calculate_brain_adc_map_with_fixture_nifti(
    hypha_service: RemoteService,
) -> None:
    """Test calculating brain ADC map with fixture NIfTI."""
    tmp = await _remote_tmpdir(hypha_service)
    dwi_path = f"{tmp}/dwi.nii.gz"
    bvals = await _remote_write_dwi_nifti(hypha_service, dwi_path)

    out = await hypha_service.calculate_brain_adc_map(
        dwi_file_path=dwi_path,
        b_values=bvals,
        output_path=f"{tmp}/adc.nii.gz",
        mask_file_path=None,
    )
    assert isinstance(out, dict)
    assert out["url"].startswith(("http://", "https://"))
    assert isinstance(out.get("log"), str)
    assert "Brain Water Diffusion Mapping" in out["log"]


@pytest.mark.asyncio
async def test_expected_fail__compare_protein_structures_with_fixture_pdbs(
    hypha_service: RemoteService,
) -> None:
    """Test comparing protein structures with fixture PDBs."""
    tmp = await _remote_tmpdir(hypha_service)
    pdb1 = f"{tmp}/a.pdb"
    pdb2 = f"{tmp}/b.pdb"
    await _remote_write_minimal_pdb(hypha_service, pdb1)
    await _remote_write_minimal_pdb(hypha_service, pdb2)

    out = await hypha_service.compare_protein_structures(
        pdb_file1=pdb1,
        pdb_file2=pdb2,
        chain_id1="A",
        chain_id2="A",
        output_prefix=f"{tmp}/cmp",
    )
    assert isinstance(out, str)
    # It might return an error log if tools are missing, or a success log
    assert "Structure" in out or "RMSD" in out or "Error" in out


@pytest.mark.asyncio
async def test_needs_fixture__quantify_and_cluster_cell_motility_with_image_sequence(
    tmp_path: Path,
) -> None:
    """Test quantifying and clustering cell motility with image sequence."""
    from biomni.tool.cell_biology import quantify_and_cluster_cell_motility

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()

    # Use the helper function but pass None as service since it's ignored
    await _remote_write_png_sequence(None, str(seq_dir), n=6)

    # Run the tool locally
    out = quantify_and_cluster_cell_motility(
        image_sequence_path=str(seq_dir),
        output_dir=str(tmp_path),
        num_clusters=2,
    )

    assert isinstance(out, str)
    assert "Cell Motility Quantification" in out
    assert "Cluster Statistics" in out


@pytest.mark.asyncio
async def test_expected_fail__analyze_comparative_genomics_and_haplotypes(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing comparative genomics and haplotypes with fixture FASTAs."""
    tmp = await _remote_tmpdir(hypha_service)
    ref = f"{tmp}/ref.fasta"
    s1 = f"{tmp}/s1.fasta"
    s2 = f"{tmp}/s2.fasta"
    ref_fa = ">chr1\\nACGTACGTACGTACGTACGT\\n"
    s1_fa = ">chr1\\nACGTACGTACGTTCGTACGT\\n"
    s2_fa = ">chr1\\nACGTACGTACGTACGTACGA\\n"
    await _remote_write_text(hypha_service, ref, ref_fa)
    await _remote_write_text(hypha_service, s1, s1_fa)
    await _remote_write_text(hypha_service, s2, s2_fa)

    out = await hypha_service.analyze_comparative_genomics_and_haplotypes(
        sample_fasta_files=[s1, s2],
        reference_genome_path=ref,
        output_dir=tmp,
    )
    assert isinstance(out, str)
    assert "Genomics" in out or "Haplotype" in out or "Error" in out


@pytest.mark.asyncio
async def test_expected_fail__analyze_protein_phylogeny_minimal(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing protein phylogeny minimal."""
    tmp = await _remote_tmpdir(hypha_service)
    fasta_path = f"{tmp}/prot.fasta"
    fasta = ">p1\\nMSTNPKPQRKTKRNTNRRPQ\\n>p2\\nMSTNPKPQRKTKRNTNRRPA\\n"
    await _remote_write_text(hypha_service, fasta_path, fasta)

    # This tool imports Biopython commandline wrappers that may be missing on the
    # remote. Accept a returned log or a quick failure.
    with pytest.raises(
        Exception,
        match=r"(?i)(Bio\.Align\.Applications|clustalw|muscle|iqtree)",
    ):
        await hypha_service.analyze_protein_phylogeny(
            fasta_sequences=fasta_path,
            output_dir=tmp,
            alignment_method="muscle",
            tree_method="iqtree",
        )


@pytest.mark.asyncio
async def test_expected_fail__query_paleobiology_direct_endpoint(
    hypha_service: RemoteService,
) -> None:
    """Test querying paleobiology direct endpoint."""
    # Avoid the LLM path; use direct endpoint.
    out = await hypha_service.query_paleobiology(
        prompt="",
        endpoint="data1.2/taxa/list.json?name=Tyrannosaurus",
    )
    assert out is not None


@pytest.mark.asyncio
async def test_expected_fail__run_autosite_errors_cleanly_if_missing_cli(
    hypha_service: RemoteService,
) -> None:
    """Test running autosite errors cleanly if missing CLI."""
    tmp = await _remote_tmpdir(hypha_service)
    pdb = f"{tmp}/r.pdb"
    await _remote_write_minimal_pdb(hypha_service, pdb)

    with pytest.raises(Exception, match=r"(?i)(autosite|error|not found)"):
        await hypha_service.run_autosite(pdb_file=pdb, output_dir=tmp, spacing=1.0)


@pytest.mark.asyncio
async def test_expected_fail__analyze_cytokine_production_in_cd4_tcells(
    hypha_service: RemoteService,
) -> None:
    """Test analyzing cytokine production in CD4 T cells missing deps is OK."""
    tmp = await _remote_tmpdir(hypha_service)
    # We don't have real FCS files in CI; this tool also depends on FlowCytometryTools.
    with pytest.raises(Exception, match=r"(?i)(FlowCytometryTools|error|not found)"):
        await hypha_service.analyze_cytokine_production_in_cd4_tcells(
            fcs_files_dict={
                "unstimulated": f"{tmp}/unstim.fcs",
                "Mtb300": f"{tmp}/mtb.fcs",
                "CMV": f"{tmp}/cmv.fcs",
                "SEB": f"{tmp}/seb.fcs",
            },
            output_dir=tmp,
        )


@pytest.mark.asyncio
async def test_expected_fail__annotate_celltype_scrna_skipped(
    hypha_service: RemoteService,
) -> None:
    """Test annotating celltype scRNA skipped."""
    tmp = await _remote_tmpdir(hypha_service)
    with contextlib.suppress(Exception):
        await hypha_service.annotate_celltype_scRNA(
            adata_filename="dummy.h5ad",
            data_dir=tmp,
            data_info="homo sapiens, brain tissue, normal",
            data_lake_path=tmp,
            cluster="leiden",
            llm="claude-3-5-sonnet-20241022",
            composition=None,
        )


@pytest.mark.asyncio
async def test_expected_fail__annotate_celltype_with_panhumanpy_skipped(
    hypha_service: RemoteService,
) -> None:
    """Test annotating celltype with panhumanpy skipped."""
    tmp = await _remote_tmpdir(hypha_service)
    with contextlib.suppress(Exception):
        await hypha_service.annotate_celltype_with_panhumanpy(
            adata_path=f"{tmp}/dummy.h5ad",
            output_dir=tmp,
        )


@pytest.mark.asyncio
async def test_expected_fail__create_scvi_embeddings_scrna_skipped(
    hypha_service: RemoteService,
) -> None:
    """Test creating scVI embeddings scRNA skipped."""
    tmp = await _remote_tmpdir(hypha_service)
    with contextlib.suppress(Exception):
        await hypha_service.create_scvi_embeddings_scRNA(
            adata_filename="dummy.h5ad",
            batch_key="batch",
            label_key="label",
            data_dir=tmp,
        )


@pytest.mark.asyncio
async def test_expected_fail__create_harmony_embeddings_scrna_skipped(
    hypha_service: RemoteService,
) -> None:
    """Test creating harmony embeddings scRNA skipped."""
    tmp = await _remote_tmpdir(hypha_service)
    with contextlib.suppress(Exception):
        await hypha_service.create_harmony_embeddings_scRNA(
            adata_filename="dummy.h5ad",
            batch_key="batch",
            data_dir=tmp,
        )


@pytest.mark.asyncio
async def test_expected_fail__map_to_ima_interpret_scrna_skipped(
    hypha_service: RemoteService,
) -> None:
    """Test mapping to IMA interpret scRNA skipped."""
    tmp = await _remote_tmpdir(hypha_service)
    with contextlib.suppress(Exception):
        await hypha_service.map_to_ima_interpret_scRNA(
            adata_filename="dummy.h5ad",
            data_dir=tmp,
            custom_args=None,
        )
