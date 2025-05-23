
* [ ] Refactor `make_plot_grid` API - Replace multiple dict-based parameters (`subplot_kwargs`, `gs_kwargs`, etc.) with a single configuration object or builder pattern to improve discoverability and reduce signature complexity.
* [ ] Add full type annotations to public APIs - Annotate return types and parameter types on functions like `make_plot_grid` so that static analysis tools can catch mismatches and developers can understand interfaces at a glance.
* [ ] Expand `make_plot_grid` test parameters - Parameterize additional combinations of `sharex`, `sharey`, and `sharez` (including mixed and edge cases) to improve coverage of axis-sharing logic.
