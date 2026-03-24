TXT_COLS_TRAIN = [
    "transaction type",
    "payee",
    "transaction detail",
]

LABEL_COL = "gl code"

TXT_COLS_TEST = [
    "transaction type",
    "payee",
    "transaction detail",
]

# Name-based field encoding: derived from the full Jan-Dec 2025 dataset.
# Null -> 0, known name -> its ID, unknown name -> max(ID) + 1.
NAME_TO_ID = {
    "angelika jamal": 1,
    "cory eden": 2,
    "dillon hall": 3,
    "emilio lagman": 4,
    "emily gardner": 5,
    "erin baile": 6,
    "harjot kaur": 7,
    "hector alfonso pertierra marin": 8,
    "janessa ellis": 9,
    "joyce pong": 10,
    "karina plazola": 11,
    "kayla symes": 12,
    "michael galpin": 13,
    "miko bacomo": 14,
    "tim morrison": 15,
    "victor ko": 16,
    "vishnu gattu": 17,
}
NAME_UNKNOWN_ID = max(NAME_TO_ID.values()) + 1  # 18
