# Copyright 2025 Kaggle Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# ---
# title: "Output types: Cryptography"
# ---
# %%

import random
import string

from kaggle_benchmarks import actors, benchmark, llm, task, utils

random.seed(0)


def substitution_cipher(text, key):
    mapping = str.maketrans(alphabet, key)
    encrypted_text = text.translate(mapping)
    return encrypted_text


def levenshtein_distance(s1, s2):
    """Calculates the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m = len(s1)
    n = len(s2)

    previous_row = list(range(n + 1))

    for i in range(m):
        current_row = [i + 1] + [0] * n

        for j in range(n):
            current_row[j + 1] = min(
                previous_row[j + 1] + 1,
                current_row[j] + 1,
                previous_row[j] + (s1[i] != s2[j]),
            )

        previous_row = current_row

    return previous_row[n]


LOWERCASE_ALPHABET = string.ascii_lowercase


def caesar_cipher(char, shift):
    return "".join(
        LOWERCASE_ALPHABET[(ord(x) + shift - ord("a")) % 26]
        if x in LOWERCASE_ALPHABET
        else x
        for x in char
    )


def string_diff(x, y):
    distance = levenshtein_distance(x, y)
    if distance:
        message = f"Distance: {distance}\n{utils.string_diff_as_markdown(x, y)}"
    else:
        message = "Strings are identical"

    actors.assertion.send(message)
    return distance


# %%


@task()
def ceasar_decryption(llm, plaintext, shift) -> int:
    encrypted = caesar_cipher(plaintext, shift)
    decrypted = llm.prompt(
        f"The text was encyrpted using caesar cypher with shift {shift}. Can you decipher it? `{encrypted}`.\nOutput the plaintext only."
    ).strip()
    return string_diff(plaintext, decrypted)


text = "the melted clocks chime for no one"
ceasar_decryption.run(llm, plaintext=text, shift=3)
# %%
text = "a fountain signed by r. mutt was lost"
ceasar_decryption.run(llm, plaintext=text, shift=3)
# %%
alphabet = string.ascii_lowercase
key = "".join(random.sample(alphabet, len(alphabet)))
text = "a cloud in trousers strolls the boulevard"


@task()
def substitution_encryption(llm, plaintext: str, key: str) -> int:
    """Checks LLM's ability to encrypt text using a substitution cipher."""
    table = ", ".join((f"{k} -> {v}" for k, v in zip(alphabet, key)))
    ciphertext = substitution_cipher(plaintext, key)
    actors.system.send("Output encrypted text only.")
    answer = llm.prompt(
        f"Help me encrypt message `{plaintext}` using a substitution cipher with the following key: {table}"
    )
    return string_diff(ciphertext, answer)


substitution_encryption.run(llm, plaintext=text, key=key)
# %%


@task()
def substitution_decryption(llm, plaintext: str, key: str) -> int:
    """Checks LLM's ability to decrypt text using a substitution cipher."""
    table = ", ".join((f"{k} -> {v}" for k, v in zip(alphabet, key)))
    ciphertext = substitution_cipher(plaintext, key)

    actors.system.send("Output palintext only.")
    answer = llm.prompt(
        f"Help me decrypt message `{ciphertext}` that was encrypted using a substitution cipher with the following key: {table}"
    )

    return string_diff(plaintext, answer)


substitution_decryption.run(llm, plaintext="this is not a pipe or a teapot", key=key)

# %%


@benchmark()
def cryptography(llm, texts, seed=0) -> int:
    """
    Checks LLMs ability to encrypt/decrypt text using various ciphers.
    """
    random.seed(seed)

    substitution_key = "".join(random.sample(alphabet, len(alphabet)))

    total = 0
    for plaintext in texts:
        total += ceasar_decryption.run(llm, plaintext=plaintext, shift=3).result
        total += substitution_encryption.run(
            llm, plaintext=plaintext, key=substitution_key
        ).result
        total += substitution_decryption.run(
            llm, plaintext=plaintext, key=substitution_key
        ).result

    return total


# %%

cryptography.run(
    llm,
    texts=[
        "meet you on a dissecting table. umbrella",
        "the marvellous is always beautiful",
    ],
)
# %%
