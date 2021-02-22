#!/usr/bin/env bash
cd ..

# Run main systems and evaluate its variants
python3 main.py --project_dir "/mnt/sda4/hcs-datasets/libssh@libssh-mirror" --considers_variants 'true' --output_dir 'results/experiments_deterministic_libssh/' --datasets 'libssh@total' 

# For each variant
python3 main.py --project_dir "/mnt/sda4/hcs-datasets/libssh@libssh-mirror" --output_dir 'results/experiments_deterministic_libssh/' --datasets 'libssh@CentOS7-openssl'  'libssh@CentOS7-openssl 1_0_x-x86-64'  'libssh@Debian-openssl 1_0_x-aarch64'  'libssh@Debian_cross_mips-linux-gnu' 'libssh@Fedora-libgcrypt-x86-64' 'libssh@Fedora-openssl' 'libssh@Fedora-openssl 1_1_x-x86-64' 'libssh@address-sanitizer' 'libssh@centos7-openssl_1_0_x-x86-64' 'libssh@centos7-openssl_1_0_x-x86_64' 'libssh@fedora-libgcrypt-x86-64' 'libssh@fedora-libgcrypt-x86_64' 'libssh@fedora-mbedtls-x86-64' 'libssh@fedora-mbedtls-x86_64' 'libssh@fedora-openssl_1_1_x-x86-64' 'libssh@fedora-openssl_1_1_x-x86-64-release' 'libssh@fedora-openssl_1_1_x-x86_64' 'libssh@fedora-openssl_1_1_x-x86_64-fips' 'libssh@fedora-openssl_1_1_x-x86_64-minimal' 'libssh@fedora-undefined-sanitizer' 'libssh@freebsd-x86_64' 'libssh@mingw32' 'libssh@mingw64' 'libssh@pages' 'libssh@tumbleweed-openssl_1_1_x-x86-64' 'libssh@tumbleweed-openssl_1_1_x-x86-64-release' 'libssh@tumbleweed-openssl_1_1_x-x86_64-clang' 'libssh@tumbleweed-openssl_1_1_x-x86_64-gcc' 'libssh@tumbleweed-openssl_1_1_x-x86_64-gcc7' 'libssh@tumbleweed-undefined-sanitizer' 'libssh@ubuntu-openssl_1_1_x-x86_64' 'libssh@undefined-sanitizer' 'libssh@visualstudio-x86' 'libssh@visualstudio-x86_64'  
 