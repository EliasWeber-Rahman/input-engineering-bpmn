from typing import List


#Levenshtein
def token_edit_distance(tokens_a: List[str], tokens_b: List[str]) -> int:
    length_a = len(tokens_a)
    length_b = len(tokens_b)

    previous_row = list(range(length_b + 1))
    for index_a in range(1, length_a + 1):
        current_row = [index_a] + [0] * length_b
        for index_b in range(1, length_b + 1):
            if tokens_a[index_a - 1] == tokens_b[index_b - 1]:
                current_row[index_b] = previous_row[index_b - 1]
            else:
                current_row[index_b] = 1 + min(
                    previous_row[index_b],
                    current_row[index_b - 1],
                    previous_row[index_b - 1],
                )
        previous_row = current_row
    return previous_row[length_b]


def edit_rate(original: str, noised: str) -> float:
    original_tokens = original.split()
    noised_tokens = noised.split()
    if not original_tokens:
        return 0.0
    distance = token_edit_distance(original_tokens, noised_tokens)
    return distance / len(original_tokens)
