"""Current Lisa component folders, shared by generation and packaging."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ASSEMBLY_MODEL=ROOT/'assembly/output/model'
ASSEMBLY_RENDER=ROOT/'assembly/output/render'
REFERENCE=ASSEMBLY_MODEL/'reference'
SUPPLIER_REFERENCE=REFERENCE/'supplier'


def part_folder(stem):
    for prefix,folder in [('motor_','motor holder'),('encoder_','encoder holder'),
                          ('shaft_','shaft frame'),('gear_','gear adapter')]:
        if stem.startswith(prefix):return ROOT/folder
    raise ValueError(f'No component folder assigned to {stem}')
