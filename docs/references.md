# References

Cite by number. Every entry resolves by DOI; no PDFs are tracked here.

1. P. M. W. Gill, M. Head-Gordon, and J. A. Pople, "An efficient algorithm for
   the generation of two-electron repulsion integrals over Gaussian basis
   functions," International Journal of Quantum Chemistry **36**, 269–280
   (1989). doi:10.1002/qua.560360831
2. P. M. W. Gill, M. Head-Gordon, and J. A. Pople, "Efficient computation of
   two-electron-repulsion integrals and their nth-order derivatives using
   contracted Gaussian basis sets," The Journal of Physical Chemistry **94**,
   5564–5572 (1990). doi:10.1021/j100377a031
3. P. M. W. Gill, B. G. Johnson, and J. A. Pople, "Two-electron repulsion
   integrals over Gaussian s functions," International Journal of Quantum
   Chemistry **40**, 745–752 (1991). doi:10.1002/qua.560400604
4. P. M. W. Gill and J. A. Pople, "The prism algorithm for two-electron
   integrals," International Journal of Quantum Chemistry **40**, 753–772
   (1991). doi:10.1002/qua.560400605
5. B. G. Johnson, P. M. W. Gill, and J. A. Pople, "Exact and approximate
   solutions to the one-center McMurchie–Davidson tree-search problem,"
   International Journal of Quantum Chemistry **40**, 809–827 (1991).
   doi:10.1002/qua.560400610
6. B. G. Johnson, P. M. W. Gill, and J. A. Pople, "The efficient transformation
   of (m0|n0) to (ab|cd) two-electron repulsion integrals," Chemical Physics
   Letters **206**, 229–238 (1993). doi:10.1016/0009-2614(93)85546-Z
7. P. M. W. Gill, "Molecular integrals over Gaussian basis functions," Advances
   in Quantum Chemistry **25**, 141–205 (1994).
   doi:10.1016/S0065-3276(08)60019-2
8. T. R. Adams, R. D. Adamson, and P. M. W. Gill, "A tensor approach to
   two-electron matrix elements," The Journal of Chemical Physics **107**,
   124–131 (1997). doi:10.1063/1.474359
9. J. T. Fermann and E. F. Valeev, "Fundamentals of molecular integrals
   evaluation," arXiv:2007.12057 [quant-ph] (2020).
   doi:10.48550/arXiv.2007.12057 — `[FV]`. Openly readable; equation
   numbers cited against this entry are the arXiv v1's. Cited by
   `docs/implementation/rys_fock.md` for eqs. 6.4–6.5 (`F_m` and the
   `(ss|ss)` prefactor), 7.3 and 7.9–7.12 (the Rys measure and rule) and
   7.14 / 7.23 (the Obara–Saika VRR the 2D recurrence is read off). It
   prints no Rys recurrence of its own and cites entry 49 for them.
10. C. C. Forgy and D. A. Mazziotti, "A differential derivation of the
    Obara–Saika relation for Gaussian electron repulsion integrals,"
    arXiv:2607.00102 [physics.chem-ph] (2026).
    doi:10.48550/arXiv.2607.00102
11. M. J. Frisch, B. G. Johnson, P. M. W. Gill, D. J. Fox, and R. H. Nobes, "An
    improved criterion for evaluating the efficiency of two-electron integral
    algorithms," Chemical Physics Letters **206**, 225–228 (1993).
    doi:10.1016/0009-2614(93)85545-Y
12. B. G. Johnson, P. M. W. Gill, J. A. Pople, and D. J. Fox, "Computing
    molecular electrostatic potentials with the PRISM algorithm," Chemical
    Physics Letters **206**, 239–246 (1993).
    doi:10.1016/0009-2614(93)85547-2
13. P. M. W. Gill, B. G. Johnson, and J. A. Pople, "A simple yet powerful upper
    bound for Coulomb integrals," Chemical Physics Letters **217**, 65–68
    (1994). doi:10.1016/0009-2614(93)E1340-M
14. G. Beylkin and S. Sharma, "A fast algorithm for computing the Boys
    function," The Journal of Chemical Physics **155**, 174117 (2021).
    doi:10.1063/5.0062444
15. D. N. Laikov, "Global approximation to the Boys functions for vectorized
    computation," The Journal of Chemical Physics **163**, 024104 (2025).
    doi:10.1063/5.0275063
16. R. Vikhamar-Sandberg and M. Repisky, "Efficient Boys function evaluation
    using minimax approximation," The Journal of Chemical Physics **164**,
    154111 (2026). doi:10.1063/5.0316073
17. A. Asadchev and E. F. Valeev, "Memory-efficient recursive evaluation of
    3-center Gaussian integrals," Journal of Chemical Theory and Computation
    **19**, 1698–1710 (2023). doi:10.1021/acs.jctc.2c00995
18. D. B. Williams-Young, A. Asadchev, D. T. Popovici, D. Clark, J. Waldrop,
    T. L. Windus, E. F. Valeev, and W. A. de Jong, "Distributed memory, GPU
    accelerated Fock construction for hybrid, Gaussian basis density functional
    theory," The Journal of Chemical Physics **158**, 234104 (2023).
    doi:10.1063/5.0151070
19. A. Asadchev and E. F. Valeev, "High-performance evaluation of high angular
    momentum 4-center Gaussian integrals on modern accelerated processors," The
    Journal of Physical Chemistry A **127**, 10889–10895 (2023).
    doi:10.1021/acs.jpca.3c04574
20. A. Asadchev and E. F. Valeev, "3-center and 4-center 2-particle Gaussian AO
    integrals on modern accelerated processors," The Journal of Chemical Physics
    **160**, 244109 (2024). doi:10.1063/5.0217001
21. A. Asadchev and E. F. Valeev, "Implementation of McMurchie–Davidson
    algorithm for Gaussian AO integrals suited for SIMD processors," The Journal
    of Physical Chemistry A **129**, 9788–9797 (2025).
    doi:10.1021/acs.jpca.5c04136
22. R. Stocks, E. Palethorpe, and G. M. J. Barca, "Multi-GPU RI-HF energies and
    analytic gradients — toward high-throughput ab initio molecular dynamics,"
    Journal of Chemical Theory and Computation **20**, 7503–7515 (2024).
    doi:10.1021/acs.jctc.4c00877
23. E. Palethorpe, R. Stocks, and G. M. J. Barca, "Advanced techniques for
    high-performance Fock matrix construction on GPU clusters," Journal of
    Chemical Theory and Computation **20**, 10424–10442 (2024).
    doi:10.1021/acs.jctc.4c00994 — `[Fock]`
24. Y. Wang, D. Hait, K. G. Johnson, O. J. Fajen, J. H. Zhang, R. D. Guerrero,
    and T. J. Martínez, "Extending GPU-accelerated Gaussian integrals in the
    TeraChem software package to f type orbitals: Implementation and
    applications," The Journal of Chemical Physics **161**, 174118 (2024).
    doi:10.1063/5.0233523
25. Y. Wang, D. Hait, P. A. Unzueta, J. H. Zhang, and T. J. Martínez, "Fast and
    scalable GPU-accelerated quantum chemistry for periodic systems with
    Gaussian orbitals: Implementation and hybrid density functional theory
    calculations," arXiv:2410.22278 [physics.chem-ph] (2024).
    doi:10.48550/arXiv.2410.22278
26. R. Li, Q. Sun, X. Zhang, and G. K.-L. Chan, "Introducing GPU acceleration
    into the Python-based simulations of chemistry framework," The Journal of
    Physical Chemistry A **129**, 1459–1468 (2025).
    doi:10.1021/acs.jpca.4c05876 — `[GPU4PySCF]`. Also arXiv:2407.09700
    (v1, openly readable); equation numbers cited against this entry are the
    arXiv v1's, which the journal version may renumber. Cited by
    `docs/implementation/rys_fock.md` for its eq. 1 (the ERI as
    `SUM_n w_n I_x I_y I_z`) and eq. 3 (`N = floor(L/2) + 1` roots); its
    Alg. 2 is not leaned on.
27. X. Wu, Q. Sun, Z. Pu, T. Zheng, W. Ma, W. Yan, Y. Xia, Z. Wu, M. Huo,
    X. Li, W. Ren, S. Gong, Y. Zhang, and W. Gao, "Enhancing GPU-acceleration
    in the Python-based simulations of chemistry frameworks," WIREs
    Computational Molecular Science **15**, e70008 (2025).
    doi:10.1002/wcms.70008
28. M. Manathunga, Y. Miao, D. Mu, A. W. Götz, and K. M. Merz, Jr., "Parallel
    implementation of density functional theory methods in the Quantum
    Interaction Computational Kernel program," Journal of Chemical Theory and
    Computation **16**, 4315–4326 (2020). doi:10.1021/acs.jctc.0c00290
29. M. Manathunga, C. Jin, V. W. D. Cruzeiro, Y. Miao, D. Mu, K. Arumugam,
    K. Keipert, H. M. Aktulga, K. M. Merz, Jr., and A. W. Götz, "Harnessing the
    power of multi-GPU acceleration into the Quantum Interaction Computational
    Kernel program," Journal of Chemical Theory and Computation **17**,
    3955–3966 (2021). doi:10.1021/acs.jctc.1c00145
30. M. Hernández Vera, "FSIM: A pedagogical and extensible HPC framework for
    the Hartree–Fock method," arXiv:2510.24968 [physics.chem-ph] (2025).
    doi:10.48550/arXiv.2510.24968
31. K. G. Johnson, S. Mirchandaney, E. Hoag, A. Heirich, A. Aiken, and
    T. J. Martínez, "Multinode multi-GPU two-electron integrals: Code
    generation using the Regent language," Journal of Chemical Theory and
    Computation **18**, 6522–6536 (2022). doi:10.1021/acs.jctc.2c00414
32. X. Wu, Q. Sun, and Y. Wang, "Designing quantum chemistry algorithms with
    just-in-time compilation," The Journal of Chemical Physics **164**, 194102
    (2026). doi:10.1063/5.0314659
33. R. D. Guerrero, "RECURSUM: Automated code generation for recurrence
    relations exceeds expert optimization via LayeredCodegen," arXiv:2604.18585
    [cs.PL] (2026). doi:10.48550/arXiv.2604.18585
34. Y. Zhang, X. Wei, J. Chen, F. Ju, W. Hu, J. Yang, and H. Xia, "FusionRCG:
    Orchestrating recursive computation graphs across GPU memory hierarchies,"
    arXiv:2605.10312 [physics.comp-ph] (2026).
    doi:10.48550/arXiv.2605.10312
35. E. Palethorpe and G. M. J. Barca, "High-performance, high-angular-momentum
    J engine on graphics processing units," Journal of Chemical Theory and
    Computation **21**, 9388–9403 (2025). doi:10.1021/acs.jctc.5c00775 —
    `[JEngine]`. Also a ChemRxiv v2 preprint,
    doi:10.26434/chemrxiv-2025-2lntb-v2, whose equation numbering is what
    `[JEngine, v2, eq. N]` indexes; the journal version may renumber. The v1
    preprint, doi:10.26434/chemrxiv-2025-2lntb, is `[JEngine, v1, …]`.
36. P. T. Bauman et al. (31 authors), "Experiences readying applications for
    Exascale," in Proceedings of the International Conference for High
    Performance Computing, Networking, Storage and Analysis (SC '23) (ACM,
    2023), pp. 1–13. doi:10.1145/3581784.3607065
37. R. Stocks, J. L. Galvez Vallejo, F. C. Y. Yu, C. Snowdon, E. Palethorpe,
    J. Kurzak, D. Bykov, and G. M. J. Barca, "Breaking the million-electron and
    1 EFLOP/s barriers: Biomolecular-scale ab initio molecular dynamics using
    MP2 potentials," in SC24: International Conference for High Performance
    Computing, Networking, Storage and Analysis (IEEE, 2024), pp. 1–12.
    doi:10.1109/SC41406.2024.00015
38. H. Huang, W. Shao, and J. Hammond, "Accelerating density fitting with
    adaptive precision and 8-bit integer on AI accelerators," The Journal of
    Physical Chemistry A **130**, 3483–3493 (2026).
    doi:10.1021/acs.jpca.6c00225
39. L. E. McMurchie and E. R. Davidson, "One- and two-electron integrals over
    Cartesian Gaussian functions," Journal of Computational Physics **26**,
    218–231 (1978). doi:10.1016/0021-9991(78)90092-X
40. S. Obara and A. Saika, "Efficient recursive computation of molecular
    integrals over Cartesian Gaussian functions," The Journal of Chemical
    Physics **84**, 3963–3974 (1986). doi:10.1063/1.450106
41. M. Head-Gordon and J. A. Pople, "A method for two-electron Gaussian
    integral and integral derivative evaluation using recurrence relations,"
    The Journal of Chemical Physics **89**, 5777–5786 (1988).
    doi:10.1063/1.455553
42. J. Rys, M. Dupuis, and H. F. King, "Computation of electron repulsion
    integrals using the Rys quadrature method," Journal of Computational
    Chemistry **4**, 154–157 (1983). doi:10.1002/jcc.540040206
43. C. A. White and M. Head-Gordon, "A J matrix engine for density functional
    theory calculations," The Journal of Chemical Physics **104**, 2620–2629
    (1996). doi:10.1063/1.470986
44. Y. Shao and M. Head-Gordon, "An improved J matrix engine for density
    functional theory calculations," Chemical Physics Letters **323**, 425–433
    (2000). doi:10.1016/S0009-2614(00)00524-8
45. U. Ryu, M. Kim, and Y. S. Lee, "Optimal use of the recurrence relations
    for the evaluation of molecular integrals over Cartesian Gaussian basis
    functions," Journal of Computational Chemistry **14**, 30–36 (1993).
    doi:10.1002/jcc.540140107
46. L. E. McMurchie and E. R. Davidson, "One and two electron integrals over
    Cartesian Gaussian functions," University of Washington technical report
    UCRL-13791 (1976). doi:10.2172/5184486 — the authors' report of entry 39,
    and an openly readable text of it. Equation numbers cited against this
    entry are the report's; the journal version may renumber.
47. F. Weigend, "A fully direct RI-HF algorithm: Implementation, optimised
    auxiliary basis sets, demonstration of accuracy and efficiency," Physical
    Chemistry Chemical Physics **4**, 4285–4291 (2002). doi:10.1039/b204199p
    — the paper defining the cc-pVXZ JK-fitting auxiliary sets, among them
    cc-pVTZ-jkfit (the Basis Set Exchange's reference for that set, and the
    citation in PySCF's `cc-pvtz-jkfit.dat`), the auxiliary basis of the
    `h2o_ccpvtz_jkfit_metric` fixture (#411).
48. T. Helgaker, P. Jørgensen, and J. Olsen, *Molecular Electronic-Structure
    Theory* (John Wiley & Sons, Chichester, 2000). doi:10.1002/9781119019572
    — §9.5, the multipole moment integrals over Cartesian Gaussians by the
    Hermite route (the moments `M^e_t` of a Hermite Gaussian and their
    recurrence in `e`), cited for `Target.DIPOLE` (#518). The equation
    number within §9.5 was not checked against the text.
49. M. Dupuis, J. Rys, and H. F. King, "Evaluation of molecular integrals over
    Gaussian basis functions," The Journal of Chemical Physics **65**, 111–116
    (1976). doi:10.1063/1.432807 — `[DRK]`. The Rys method's 2D-integral
    recurrences and their coefficients, the source of
    `docs/implementation/rys_fock.md`'s VRR. Closed access: neither its
    equation numbers nor its own spelling of the coefficients' names was
    checked against the text here.
50. R. Lindh, U. Ryu, and B. Liu, "The reduced multiplication scheme of the
    Rys quadrature and new recurrence relations for auxiliary function based
    two-electron integral evaluation," The Journal of Chemical Physics **95**,
    5889–5897 (1991). doi:10.1063/1.461610 — `[LRL]`. The same 2D recurrences
    re-derived with a reduced operation count. Closed access; no equation
    number cited against it here was checked against the text.
51. Q. Sun, "Libcint: An efficient general integral library for Gaussian basis
    functions," Journal of Computational Chemistry **36**, 1664–1671 (2015).
    doi:10.1002/jcc.23981 — `[Libcint]`. Cited for its source, which
    implements `[DRK]`'s recurrences and is this repo's oracle (through
    PySCF): `src/g2e.c`'s `CINTg0_2e` (the coefficients) and
    `CINTg0_2e_2d` (the recurrences), and `src/rys_roots.c`'s
    `CINTrys_roots` with `src/roots_for_x0.dat`'s `POLY_LARGEX_RT` /
    `POLY_LARGEX_WW` (the large-`T` rule), read at commit `4909749` of
    github.com/sunqm/libcint. Also arXiv:1412.0649 (openly readable),
    which prints none of the recurrences and cites entry 49 for them.
