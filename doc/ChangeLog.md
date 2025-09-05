(changelog)=

# Changelog

## Version 0.9.0

* Expose non-interacting impurity Green's functions and hybridization functions
  via attributes of `EDIpackSolver`.
* Expose dynamical susceptibilities (response functions) via attributes of
  `EDIpackSolver`. The attributes are called `chi_spin_{iw,w,tau}` for the spin
  channel, `chi_dens_{iw,w,tau}` for the charge channel, `chi_pair_{iw,w,tau}`
  for the pair channel, and `chi_exct_{iw,w,tau}` for the exciton channel.

## Version 0.8.0

* Add support for calculations without bath.
* Allow for read-write access to `EDIpackSolver.hloc`.
* `EDIpackSolver.superconductive_phi` has been changed to be the modulus of the
  computed superconductive order parameter. The complex argument of the order
  parameter is now accessible as `EDIpackSolver.superconductive_phi_arg`.
* `EDIpackSolver`'s constructor got a new keyword argument `keep_dir`, which
  disables automatic deletion of EDIpacks's temporary directory upon object
  destruction. This option is ignored on Python < 3.12.
* Do not ignore the argument `lanc_nstates_sector` of `EDIpackSolver`'s
  constructor in the zero temperature mode.
* Fixed a bug in `EDIpackSolver.chi2_fit_bath()`, which made this method
  unusable without first calling `EDIpackSolver.solve()`. There is now a unit
  test that covers the bath fitting functionality.

## Version 0.7.0

* Add support for general two-particle interactions.
* Add support for zero temperature calculations.

## Version 0.6.0

Updated codebase following the naming change of EDIpack2 and EDIpy2.
C.f. https://github.com/EDIpack/EDIpack/issues/16.

## Version 0.5.0

This is the initial public release for this project.
