# Alignment with Hidden Conformity

Source reviewed: `Hidden Conformity (1).pdf`, 19 pages.

The repository terminology follows the presentation:

- Phase 1: convert an LLM into a complete local cellular-automaton rule.
- Phase 2: validate frozen LLM-induced rules on the Density Classification Task.
- Phase 3: study memory-enabled conformity dynamics with repeated LLM queries.

The current RTX 5090 memory experiments belong to Phase 3. The newly deployed rule producer and automaton consumer cover Phases 1 and 2. They are linked by an explicit rule manifest, not by shared mutable directories.
