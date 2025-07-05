#!bin/bash
python -c "import sys; sys.path.insert(0, '../'); from multistage_pipeline.utils import create_data_dirs; create_data_dirs()"