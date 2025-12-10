# Tested Biomni Tools Summary

This document lists all 50+ tools that have test coverage in the test suite.

## Literature & Information Extraction (6 tools)

1. `query_pubmed` - Query PubMed for papers
2. `query_arxiv` - Query arXiv for papers
3. `query_scholar` - Query Google Scholar
4. `search_google` - Google search functionality
5. `extract_url_content` - Extract webpage content
6. `fetch_supplementary_info_from_doi` - Fetch supplementary materials from DOI

## Genetics & Molecular Biology (8 tools)

1. `annotate_open_reading_frames` - Find ORFs in DNA sequences
2. `get_gene_coding_sequence` - Retrieve gene coding sequences
3. `align_sequences` - Align primer sequences
4. `find_sequence_mutations` - Identify mutations between sequences
5. `design_primer` - Design PCR primers
6. `find_restriction_enzymes` - Find restriction enzyme sites
7. `digest_sequence` - Simulate restriction enzyme digestion
8. `pcr_simple` - Simulate PCR amplification

## Cell Biology & Microscopy (7 tools)

1. `analyze_cell_morphology_and_cytoskeleton` - Analyze cell morphology
2. `quantify_cell_cycle_phases_from_microscopy` - Quantify cell cycle phases
3. `analyze_mitochondrial_morphology_and_potential` - Analyze mitochondria
4. `analyze_protein_colocalization` - Analyze protein colocalization
5. `segment_and_quantify_cells_in_multiplexed_images` - Segment multiplexed images
6. `perform_facs_cell_sorting` - FACS cell sorting simulation
7. `analyze_cfse_cell_proliferation` - Analyze CFSE proliferation

## Biochemistry & Structural Biology (6 tools)

1. `analyze_circular_dichroism_spectra` - Analyze CD spectroscopy
2. `analyze_enzyme_kinetics_assay` - Analyze enzyme kinetics
3. `analyze_protease_kinetics` - Analyze protease kinetics
4. `analyze_protein_conservation` - Analyze protein conservation
5. `predict_protein_disorder_regions` - Predict intrinsically disordered regions
6. `calculate_physicochemical_properties` - Calculate drug properties

## Synthetic Biology & CRISPR (7 tools)

1. `design_knockout_sgrna` - Design sgRNA for CRISPR knockout
2. `perform_crispr_cas9_genome_editing` - Simulate CRISPR editing
3. `analyze_crispr_genome_editing` - Analyze CRISPR results
4. `get_golden_gate_assembly_protocol` - Golden Gate assembly protocol
5. `design_golden_gate_oligos` - Design Golden Gate oligos
6. `get_bacterial_transformation_protocol` - Transformation protocol
7. `get_oligo_annealing_protocol` - Oligo annealing protocol

## Genomics & Bioinformatics (8 tools)

1. `gene_set_enrichment_analysis` - Gene set enrichment analysis
2. `get_gene_set_enrichment_analysis_supported_database_list` - List enrichment databases
3. `get_rna_seq_archs4` - Fetch RNA-seq data from ARCHS4
4. `perform_chipseq_peak_calling_with_macs2` - ChIP-seq peak calling
5. `find_enriched_motifs_with_homer` - Find enriched motifs
6. `analyze_chromatin_interactions` - Analyze chromatin interactions
7. `identify_transcription_factor_binding_sites` - Identify TF binding sites
8. `simulate_demographic_history` - Simulate demographic history

## Single-Cell & Systems Biology (5 tools)

1. `annotate_celltype_scRNA` - Annotate cell types in scRNA-seq
2. `create_scvi_embeddings_scRNA` - Create scVI embeddings
3. `create_harmony_embeddings_scRNA` - Create Harmony embeddings
4. `simulate_whole_cell_ode_model` - Simulate whole-cell ODE model
5. `perform_flux_balance_analysis` - Flux balance analysis

## Database Query Tools (8 tools)

1. `query_uniprot` - Query UniProt database
2. `query_pdb` - Query PDB database
3. `query_kegg` - Query KEGG database
4. `query_ensembl` - Query Ensembl database
5. `query_clinvar` - Query ClinVar database
6. `blast_sequence` - BLAST sequence search
7. `list_mirdb_species` - List miRDB species
8. `get_targets_for_mirna` - Get targets for miRNA

## Immunology & Pathology (4 tools)

1. `isolate_purify_immune_cells` - Simulate immune cell isolation
2. `analyze_flow_cytometry_immunophenotyping` - Analyze flow cytometry
3. `analyze_cell_senescence_and_apoptosis` - Analyze senescence and apoptosis
4. `analyze_immunohistochemistry_image` - Analyze IHC images

## Pharmacology & Drug Discovery (7 tools)

1. `docking_autodock_vina` - Molecular docking with AutoDock Vina
2. `predict_admet_properties` - Predict ADMET properties
3. `predict_binding_affinity_protein_1d_sequence` - Predict binding affinity
4. `query_drug_interactions` - Query drug interactions
5. `check_drug_combination_safety` - Check drug combination safety
6. `retrieve_topk_repurposing_drugs_from_disease_txgnn` - Drug repurposing candidates
7. `analyze_xenograft_tumor_growth_inhibition` - Analyze xenograft tumor growth

## Microbiology (4 tools)

1. `count_bacterial_colonies` - Count bacterial colonies
2. `analyze_bacterial_growth_curve` - Analyze bacterial growth
3. `segment_and_analyze_microbial_cells` - Segment microbial cells
4. `quantify_biofilm_biomass_crystal_violet` - Quantify biofilm biomass

## Total Coverage

**70 tool tests** covering **50+ unique Biomni tools** across 10 major biological domains.

## Test Organization

- Each category has its own test file in `tests/`
- Tests use shared fixtures from `conftest.py`
- All tests are async-compatible with pytest-asyncio
- Linting configured for test-specific patterns in `ruff.toml`
