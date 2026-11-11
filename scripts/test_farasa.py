
import os
import subprocess
import tempfile
from pathlib import Path

FARASA_JAR = Path(r"C:\Farasa\Farasa.jar")
FARASA_DATA = Path(r"C:\Farasa\FarasaData")
FARASA_LIB = Path(r"C:\Farasa\lib")

os.environ["FarasaDataDir"] = str(FARASA_DATA.resolve()) + os.sep

texts = [
    "أشعر بالسعادة والفرح عندما أحقق أهدافي.",
    "أشعر بالحزن والخوف عندما أفقد شخصاً عزيزاً.",
    "أشعر بالغضب عندما يتعرض الناس للظلم."
]

with tempfile.TemporaryDirectory() as temp_dir:

    input_file = Path(temp_dir) / "input.txt"
    output_file = Path(temp_dir) / "output.txt"

    input_file.write_text(
        "\n".join(texts) + "\n",
        encoding="utf-8"
    )

    command = [
        "java",
        "-Dfile.encoding=UTF-8",
        "-cp",
        (
            f"{FARASA_JAR};"
            f"{FARASA_LIB / 'commons-lang3-3.1.jar'};"
            f"{FARASA_LIB / 'mapdb-2.0.0-SNAPSHOT.jar'}"
        ),
        "farasa.TestCase",
        "-i",
        str(input_file),
        "-o",
        str(output_file)
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    print(result.stdout)

    if not output_file.exists():
        raise RuntimeError("Farasa output file was not created.")

    segmented = output_file.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(segmented) != len(texts):
        raise ValueError("Input/output record count mismatch.")

    print("\nFarasa Test Results:")

    for original, processed in zip(texts, segmented):
        print("\nOriginal:", original)
        print("Segmented:", processed)

print("\nFARASA TEST COMPLETED SUCCESSFULLY")
