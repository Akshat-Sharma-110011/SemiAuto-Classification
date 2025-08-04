try:
    from mealpy.human_based import (
        BRO, BSO, CA, CHIO, FBIO, GSKA, HBO, HCO,
        ICA, LCO, QSA, SARO, SPBO, SSDO, TLO
    )
    from mealpy.physics_based import (
        ASO, ArchOA, CDO, EFO, EO, EVO, FLA,
        HGSO, MVO, NRO, RIME, SA, TWO, WDO
    )
    from mealpy.math_based import (
        AOA, CEM, CGO, CircleSA, GBO, HC, INFO, PSS,
        RUN, SCA, SHIO, TS
    )
    from mealpy.evolutionary_based import (
        EP,  # BaseEP, LevyEP
        ES,  # BaseES, LevyES
        MA,  # Memetic Algorithm
        GA,  # Genetic Algorithm
        DE  # Differential Evolution and variants (JADE, SADE, SHADE…)
    )
    from mealpy.swarm_based import (
        ABC, ACOR, AGTO, ALO, AO, ARO, AVOA, BA, BES, BFO, BSA, BeesA,
        COA, CSA, CSO, CoatiOA, DMOA, DO, EHO, ESOA, FA, FFA, FFO, FOA,
        FOX, GJO, GOA, GTO, GWO, HBA, HGS, HHO, JA, MFO, MGO, MPA, MRFO,
        MSA, NGO, NMRA, OOA, PFA, POA, PSO, SCSO, SFO, SHO, SLO, SRSR,
        SSA, SSO, SSpiderA, SSpiderO, STO, SeaHO, ServalOA, TDO, TSO,
        WOA, WaOA, ZOA,
    )

    print("MEALPY is working")
except ImportError as e:
    print(f"MEALPY import failed: {e}")
