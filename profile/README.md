# Repositories

## Personal projects
- [agjax](https://github.com/mfschubert/agjax): wrapper for autograd functions that allows use with jax, including support for jax transformations such as `jit` and `vmap`.
- [fmmax](https://github.com/mfschubert/agjax): a jax implementation of the Fourier Modal Method (aka Rigorous Coupled Wave Analysis, RCWA). This repo is hard fork of the [fmmax repo](https://github.com/facebookresearch/fmmax) which I originally created while working at Meta Reality Labs.
- [jeig](https://github.com/mfschubert/jeig): nonsymmetric eigendecomposition for jax with multiple backends. Can provide significant speedup over `jnp.linalg.eig` when performing a batch of eigendecompositions.
- [mewtax](https://github.com/mfschubert/mewtax): differentiable minimization in jax using Newton's method, following the [implicit layers tutorial](https://implicit-layers-tutorial.org/implicit_functions/).
- [refractiveindex2](https://github.com/mfschubert/sparsejac): a python interface to the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database) of optical material properties, with api modeled after the [refractiveindex package](https://github.com/polyanskiy/refractiveindex.info-database).
- [sparsejac](https://github.com/mfschubert/sparsejac): enables efficient calculation of sparse jacobians using jax.


## invrs-io projects
- [invrs-gym](https://github.com/invrs-io/gym): A collection of optics design challenges with a common API, intended to facilitate research and development of new design methods. For optics designers, the gym also serves as an example of how to structure design challenges so that they can be used with other parts of the invrs.io ecosystem.
- [invrs-opt](https://github.com/invrs-io/opt): Optimization algorithms with a common API.
- [invrs-utils](https://github.com/invrs-io/utils): Utilities, including those which simplify the running and analysis of experiments.
- [leaderboard](https://github.com/invrs-io/leaderboard): A dataset of solutions to gym challenges.
- [totypes](https://github.com/invrs-io/totypes): Defines custom types used in the gym, and generally applicable to AI-guided design, topology optimization, and inverse design.


## Projects I contribute to
- [imageruler](https://github.com/nanocomp/imageruler): package to measure minimum solid and void length scales in binary images, useful as an independent metric for assessing the results of length-scale-constrained topology optimization.
- [photonics-opt-testbed](https://github.com/nanocomp/photonics-opt-testbed): a collection of inverse design challenges, as described in the paper _[Validation and characterization of algorithms and software for photonics inverse design](https://opg.optica.org/josab/abstract.cfm?uri=josab-41-2-A161)_.
