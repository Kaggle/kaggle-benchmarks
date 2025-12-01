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
# title: Calendar conversion
# ---
# %%

from kaggle_benchmarks import assertions, llm, task


@task()
def julian_to_gregorian_conversion(
    llm, julian_date: str, gregorian_date: str, desc: str
):
    response = llm.prompt(
        f"Hey, if it's {julian_date} in Julian calendar, what's the Gregorian date? Use ISO format",
    )
    assertions.assert_in(
        gregorian_date,
        response,
        (
            f"Julian to Gregorian conversion for {desc} ({julian_date}) failed.\n"
            f"Expected '{gregorian_date}', but got: {response}"
        ),
    )


@task()
def gregorian_to_julian_conversion(
    llm, gregorian_date: str, julian_date: str, desc: str
):
    response = llm.prompt(
        f"If today is {gregorian_date} in Gregorian calendar, what would be the Julian date? Use ISO format",
    )
    assertions.assert_in(
        julian_date,
        response,
        (
            f"Gregorian to Julian conversion for {desc} ({gregorian_date}) failed.\n"
            f"Expected '{julian_date}', but got: {response}"
        ),
    )


@task(name="Calendar Conversion")
def calendar_conversion(llm, dates):
    """
    Benchmark to evaluate the LLM's ability to perform Julian to Gregorian calendar conversions.

    """
    for gregorian_date, julian_date, desc in dates:
        julian_to_gregorian_conversion.run(llm, julian_date, gregorian_date, desc)
        gregorian_to_julian_conversion.run(llm, gregorian_date, julian_date, desc)


dates = [
    ("2025-02-26", "2025-02-13", "Present days"),
    ("1989-11-09", "1989-10-27", "Fall of Berlin Wall"),
    ("1789-07-14", "1789-07-03", "Storming of the Bastille"),
    # ("1582-10-05", "1582-10-15", "Gregorian reform start"),
    ("2024-02-28", "2024-02-15", "Feb 28 leap year"),
    ("2024-03-01", "2024-02-17", "Mar 1 leap year"),
    ("2023-02-28", "2023-02-15", "Feb 28 non-leap year"),
    ("2023-03-01", "2023-02-16", "Mar 1 non-leap year"),
    ("2400-02-24", "2400-02-08", "Far future date"),
]


calendar_conversion.run(llm, dates)
# %%
