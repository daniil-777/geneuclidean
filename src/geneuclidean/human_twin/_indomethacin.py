"""Plasma indomethacin measurements from Kwan et al. (1976), Table I.

Transcribed from the medicaldata package's Indometh.rda. Subject identifiers
are the original factor labels. One microgram per milliliter equals one mg/L.
"""

DATA_LICENSE_NOTICE = """MIT License

Copyright (c) 2021 medicaldata authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

SOURCE = "https://doi.org/10.1007/BF01063617"
DATA_SOURCE = (
    "https://github.com/higgi13425/medicaldata/blob/"
    "cf7eea82dad7c30ea6d62efc7bad7d01158e2bf7/data/Indometh.rda"
)
DATA_SHA256 = "6e4698c1fc777825194d1d4dfa8363cf58c64625cd4f3060ac6039b61d88c67c"
TIME_H = (0.25, 0.50, 0.75, 1.00, 1.25, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00)
SUBJECTS = (1, 2, 3, 4, 5, 6)
PLASMA_MG_L = (
    (1.50, 0.94, 0.78, 0.48, 0.37, 0.19, 0.12, 0.11, 0.08, 0.07, 0.05),
    (2.03, 1.63, 0.71, 0.70, 0.64, 0.36, 0.32, 0.20, 0.25, 0.12, 0.08),
    (2.72, 1.49, 1.16, 0.80, 0.80, 0.39, 0.22, 0.12, 0.11, 0.08, 0.08),
    (1.85, 1.39, 1.02, 0.89, 0.59, 0.40, 0.16, 0.11, 0.10, 0.07, 0.07),
    (2.05, 1.04, 0.81, 0.39, 0.30, 0.23, 0.13, 0.11, 0.08, 0.10, 0.06),
    (2.31, 1.44, 1.03, 0.84, 0.64, 0.42, 0.24, 0.17, 0.13, 0.10, 0.09),
)
