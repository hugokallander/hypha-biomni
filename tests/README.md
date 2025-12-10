# Biomni Test Suite

This directory contains comprehensive tests for the Biomni tool service.

## Test Structure

Tests are organized by tool category:

- `conftest.py` - Shared pytest fixtures and configuration
- `test_literature.py` - Literature search and information extraction (6 tools)
- `test_genetics.py` - Genetics and molecular biology tools (8 tools)
- `test_cell_biology.py` - Cell biology and microscopy analysis (7 tools)
- `test_biochemistry.py` - Biochemistry and structural biology (6 tools)
- `test_synthetic_biology.py` - CRISPR and synthetic biology (7 tools)
- `test_genomics.py` - Genomics and bioinformatics (8 tools)
- `test_singlecell.py` - Single-cell and systems biology (5 tools)
- `test_databases.py` - Database query tools (8 tools)
- `test_immunology.py` - Immunology and pathology (4 tools)
- `test_pharmacology.py` - Drug discovery and pharmacology (7 tools)
- `test_microbiology.py` - Microbiology tools (4 tools)

### Total Coverage

70+ tool tests covering 50+ unique Biomni tools

## Running Tests

### Prerequisites

1. Install pytest and required dependencies:

```bash
pip install pytest pytest-asyncio
```

1. Set up environment variables in `.env`:

```bash
HYPHA_TOKEN=your_token_here
HYPHA_SERVER_URL=https://hypha.aicell.io  # optional, defaults to this
HYPHA_WORKSPACE=hypha-agents  # optional, defaults to this
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_literature.py
```

### Run Specific Test

```bash
pytest tests/test_literature.py::TestLiteratureTools::test_query_pubmed
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Output Capture Disabled

```bash
pytest tests/ -s
```

## Test Design

Each test:

- Uses the shared `hypha_service` fixture from `conftest.py`
- Provides reasonable parameters for the tool being tested
- Asserts that the result is not None (basic smoke test)
- Is async-compatible using `@pytest.mark.asyncio`

## Adding New Tests

To add tests for a new tool:

1. Identify the appropriate test file based on tool category
2. Add a new test method to the test class
3. Provide reasonable test parameters
4. Follow the existing pattern:

```python
async def test_my_new_tool(self, hypha_service):
    """Test description."""
    result = await hypha_service.my_tool(
        param1="value1",
        param2="value2",
    )
    assert result is not None
```

## Linting

Tests follow project linting standards with pytest-specific exceptions configured in `ruff.toml`:

- `S101` - Use of assert is allowed in tests
- `ANN001/ANN201` - Type annotations not required for test methods
- `N802` - Test naming conventions allowed

Check linting:

```bash
ruff check tests/
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines to:

- Verify tool availability and basic functionality
- Catch breaking changes in the Hypha service
- Validate tool parameter schemas
- Monitor service health

## Notes

- Tests require an active Hypha service connection
- Some tools may require specific data files or external resources
- Tests are designed as smoke tests - they verify tools respond, but don't validate detailed behavior
- For detailed functional testing, consider adding integration tests with specific assertions
