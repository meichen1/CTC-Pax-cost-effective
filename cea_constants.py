"""
Shared cost constants for the CanTreatCOVID Paxlovid cost-effectiveness analysis.

Unit costs (CAD):
  COST_PAX      – Paxlovid drug course
  COST_OUTPAT   – per outpatient / family-doctor / walk-in visit
  COST_ER       – per emergency department visit
  COST_HOSPITAL – per hospital inpatient day

Source: https://www.qch.on.ca/WithoutHealthInsuranceFees
"""

COST_PAX       = 1288.88   # Paxlovid drug per course
COST_OUTPAT    = 397.0     # per outpatient / family-doctor / walk-in visit
COST_ER        = 397.0     # per ER visit
COST_HOSPITAL  = 1270.0    # per inpatient day
