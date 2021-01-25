#!/usr/bin/env bash
cd ..

# Run main systems and evaluate its variants
python3 main.py --project_dir "/mnt/sda4/hcs-datasets/core@dune-common" --considers_variants 'true' --output_dir 'results/experiments_deterministic_dune/' --datasets 'dune@total'

# For each variant
python3 main.py --project_dir "/mnt/sda4/hcs-datasets/core@dune-common" --output_dir 'results/experiments_deterministic_dune/' --datasets 'dune@debian-11-gcc-9-17-downstream' 'dune@debian-11-gcc-9-17-downstream-dune-grid' 'dune@debian-11-gcc-9-17-python' 'dune@debian_10  gcc_c__17' 'dune@debian_10 clang-6-libcpp-17' 'dune@debian_10 clang-7-libcpp-17' 'dune@debian_10 gcc-7-14--expensive' 'dune@debian_10 gcc-7-17' 'dune@debian_10 gcc-7-17--expensive' 'dune@debian_10 gcc-8-noassert-17' 'dune@debian_11 gcc-10-20' 'dune@debian_11 gcc-9-20' 'dune@debian_8--clang' 'dune@debian_8--gcc' 'dune@debian_8-backports--clang' 'dune@debian_9 clang-3_8-14' 'dune@debian_9 gcc-6-14' 'dune@debian_9--clang' 'dune@debian_9--gcc' 'dune@ubuntu-20_04-gcc-9-17-python' 'dune@ubuntu_16_04 clang-3_8-14' 'dune@ubuntu_16_04 gcc-5-14' 'dune@ubuntu_16_04--clang' 'dune@ubuntu_16_04--gcc' 'dune@ubuntu_18_04 clang-5-17' 'dune@ubuntu_18_04 clang-6-17' 'dune@ubuntu_20_04 clang-10-20' 'dune@ubuntu_20_04 gcc-10-20' 'dune@ubuntu_20_04 gcc-9-20' 