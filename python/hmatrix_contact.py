"""Backward-compatibility alias: the module is now called ``aspher``.

The project was renamed from Hcontact to ASPHER (Accelerated SPectral and
HiERarchical contact solver); ``import hmatrix_contact`` keeps working by
resolving to the ``aspher`` extension module.
"""
import sys

import aspher

sys.modules[__name__] = aspher
