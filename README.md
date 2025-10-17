# 🦜️🔗 LangChain IBM

## Packages

This repository contains 2 package with IBM integrations with LangChain:

- [langchain-ibm](https://pypi.org/project/langchain-ibm/) integrates [IBM watsonx](https://www.ibm.com/watsonx).
- [langchain-db2](https://pypi.org/project/langchain-db2/)(will be uploaded very soon) integrates IBM Db2 database vector store and vector search.

Each of these has its own development environment.

## Contribute Code

To contribute to this project, please follow the [contributing guidelines](https://docs.langchain.com/oss/python/contributing).

## Repository Structure

If you plan on contributing to LangChain-IBM code or documentation, it can be useful to understand the high level structure of the repository.

`langchain-ibm` is organized as a [monorepo](https://en.wikipedia.org/wiki/Monorepo) that contains multiple packages.

Here's the structure visualized as a tree:

```text
.
├── libs
│   ├── ibm
│   │   ├── tests/unit_tests # Unit tests (present in each package, not shown for brevity)
│   │   ├── tests/integration_tests # Integration tests (present in each package, not shown for brevity)
│   ├── langchain-db2
```

The root directory also contains the following files:

- `pyproject.toml`: Dependencies for building and linting the docs and cookbook.
- `Makefile`: A file that contains shortcuts for building and linting the docs and cookbook.

There are other files in the root directory level, but their presence should be self-explanatory.

## Local Development Dependencies

Install development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```bash
uv sync --group lint --group typing --group test --group test_integration
```

Then verify dependency installation:

```bash
make test
```

## Formatting and Linting

### Formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run formatting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the master branch using the format_diff command:

```bash
make format_diff
```

This is especially useful when you have made changes to a subset of the project and want to ensure your changes are properly formatted without affecting the rest of the codebase.

### Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run linting for docs, cookbook and templates:

```bash
make lint
```

To run linting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make lint
```

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the lint_diff command:

```bash
make lint_diff
```

This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

## Testing

All of our packages have unit tests and integration tests, and we favor unit tests over integration tests.

Unit tests run on every pull request, so they should be fast and reliable.

Integration tests run once a day, and they require more setup, so they should be reserved for confirming interface points with external services.

### Unit Tests

Unit tests cover modular logic that does not require calls to outside APIs.

If you add new logic, please add a unit test.

In unit tests we check pre/post processing and mocking all external dependencies.

To install dependencies for unit tests:

```bash
uv sync --group test
```

To run unit tests:

```bash
make test
```

To run a specific test:

```bash
TEST_FILE=tests/unit_tests/test_imports.py make test
```

### Integration Tests

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).

If you add support for a new external API, please add a new integration test.

**Warning:** Almost no tests should be integration tests.

  Tests that require making network connections make it difficult for other developers to test the code.

  Instead favor relying on `responses` library and/or `mock.patch` to mock requests using small fixtures.

To install dependencies for integration tests:

```bash
uv sync --group test --group test_integration
```

To run integration tests:

```bash
make integration_tests
```

### Prepare environment variables for local testing

- Copy `tests/integration_tests/.env.example` to `tests/integration_tests/.env`
- Set variables in `tests/integration_tests/.env` file, e.g `WATSONX_APIKEY`

Additionally, it's important to note that some integration tests may require certain environment variables to be set, such as `PROJECT_ID`. Be sure to set any required environment variables before running the tests to ensure they run correctly.

### Coverage

Code coverage (i.e. the amount of code that is covered by unit tests) helps identify areas of the code that are potentially more or less brittle.

Coverage requires the dependencies for tests:

```bash
uv sync --group test
```

To get a report of current coverage, run the following:

```bash
make coverage
```