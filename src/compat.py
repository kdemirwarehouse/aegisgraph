"""Python 3.10 çalışma zamanı (Hugging Face Spaces) için uyumluluk katmanı.

Python 3.11'de gelen `datetime.UTC` ve `enum.StrEnum` burada tek noktadan sağlanır;
modüller bu sembolleri doğrudan `src.compat` üzerinden içe aktarır.
"""

from __future__ import annotations

import sys
from datetime import timezone
from enum import Enum

UTC = timezone.utc

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """`enum.StrEnum` davranışını taklit eder: `str(üye)` değeri döndürür."""

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, format_spec: str) -> str:
            return str(self.value).__format__(format_spec)


__all__ = ["UTC", "StrEnum"]
