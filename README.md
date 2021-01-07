[<img align="right" src="https://cdn.buymeacoffee.com/buttons/default-orange.png" width="217px" height="51x">](https://www.buymeacoffee.com/pradolima)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


# Deterministic Optimal - TCPCI


### A method to find optimal solutions for test case priotization in Continuous Integration environments

This tool allows to find optimal solutions (in a deterministic way) for traditional systems and highly configurable systems (HCSs).

![](https://img.shields.io/badge/python-3.6+-blue.svg)


# Getting started

- [Citation](#citation)
- [Installing required dependencies](#installing-required-dependencies)
- [Datasets](#datasets)
	- [HCS Dataset](#hcs-dataset)
	- [Traditional Systems](#traditional-systems)
		- [Deeplearning4j](#deeplearning4j)
		- [Druid](#druid)
- [About the files input](#about-the-files-input)
	- [Industrial Representation](#industrial-representation)
- [Using the tool](#using-the-tool)
	- [Running for a traditional system](#running-for-a-traditional-system)
	- [Running for a HCS system](#running-for-a-hcs-system)
			- [Whole Test Set Strategy](#whole-test-set-strategy)
			- [Variant Test Set Strategy](#variant-test-set-strategy)
- [References](#references)
- [contributors](#Contributors)
----------------------------------


# Citation

If this tool contributes to a project which leads to a scientific publication, I would appreciate a citation.

```

```

# Installing required dependencies

The following command allows to install the required dependencies:

```
 $ pip install -r requirements.txt
 ```

# Datasets 

The datasets used in the examples (and much more datasets) are available at [Harvard Dataverse Repository](https://dataverse.harvard.edu/dataverse/gres-ufpr). Here, we present the datasets used in this README.

## HCS Dataset

**SSH library (libssh)** is an open-source C multiplatform library implementing the SSHv2 protocol on client and server-side. This library is designed to allow remotely execute programs, transfer files, use a secure and transparent tunnel, manage public keys, and the like. Libssh is a Highly-Configurable Software Systems (HCSS) that is statically configurable with the C preprocessor. It is available at [here](https://www.libssh.org/) and hosted on [GitLab](https://gitlab.com/libssh/libssh-mirror/). The dataset from the LIBSSH system is available at [here](https://doi.org/10.7910/DVN/SSTESD), and it contains records from GitLab CI build history.

## Traditional Systems

We consider traditional systems that ones that are not HCS.

### Deeplearning4j

**Deeplearning4j** is a deep learning library for Java Virtual Machine and it is available at [here](https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j). The dataset from the Deeplearning4j system is available at [here](https://doi.org/10.7910/DVN/EVR1IU), and it contains records from Travis CI build history.


### Druid

**Druid** is a database connection pool written in Java used by Alibaba, and it is available at [here](https://github.com/alibaba/druid). The dataset from the Druid system is available at [here](https://doi.org/10.7910/DVN/7DZ6X5), and it contains records from Travis CI build history.

# About the files input

This tool considers two kind of *csv files*: **features-engineered** and **data-variants**. The second file, **data-variants.csv**, is used by the HCS, and it represents all results from all variants. The information is organized by commit and variant.

*TODO*


## Industrial dataset representation

*TODO*

<!--
Describe about the variable is_industrail_dataset inside the code.
-->

#  Using the tool

## Running for a traditional system 

🔥**Main Option #1**🔥.

To find optimal solutions for a traditional system, do:

```
python main.py --project_dir 'data' --datasets 'alibaba@druid' 'deeplearning4j@deeplearning4j' --output_dir 'results/optimal_deterministic'
```

**where:** 
- `--project_dir` is the directory that constains your system. For instance, we desire to run the algorithm for the systems that are inside the directory **data**. Please, you must to inform the complete path.
- `--datasets` is an array that represents the datasets to analyse. It's the folder name inside `--project_dir` which contains the required file inputs.
- `--output_dir` is the directory where we gonna save the results.

The another parameters available are:
- `--sched_time_ratio` tghat represents the Schedule Time Ratio, that is, time constraints that represents the time available to run the tests. **Default**: 0.1 (10%), 0.5 (50%), and 0.8 (80%) of the time available.

## Running for a HCS system

In this option, we apply two strategies to find optimal solutions for the variants: **Whole Test Set Strategy (WTS)** and **Variant Test Set Strategy (VTS)**. For more information read **Ref2** in [References](#references)

### *Whole Test Set Strategy*

🔥**Main Option #2**🔥.

**Whole Test Set Strategy (WTS)** prioritizes the test set composed by the union of the test cases of all variants. To run this strategy, do: 

```
python main.py --project_dir 'data/libssh@libssh-mirror' --considers_variants 'true' --datasets 'libssh@total' --output_dir 'results/optimal_deterministic'
```

**where:** 
- `--project_dir` is the directory that constains your system. To run the variants of the **libssh**, we created subfolders inside the **libssh@libssh-mirror** directory. In this way, **libssh@libssh-mirror** represents the *project name*. Please, you must to inform the complete path.
- `--datasets` is an array that represents the datasets to analyse. It's the folder name inside `--project_dir` which contains the required file inputs. 
- `considers_variants` is a flag to consider to prioritize the variants of the systems based on the **WTS** strategy. 
- `--output_dir` is the directory where we gonna save the results.

The another parameters available are:
- `--sched_time_ratio` tghat represents the Schedule Time Ratio, that is, time constraints that represents the time available to run the tests. **Default**: 0.1 (10%), 0.5 (50%), and 0.8 (80%) of the time available.


### *Variant Test Set Strategy* 

**Variant Test Set Strategy (VTS)** prioritizes each variant as a system, that is, treating each variant independently. To run this strategy is similar to **Main Option #1**, do: 

```
python main.py --project_dir "data/libssh@libssh-mirror" --datasets 'libssh@CentOS7-openssl' 'libssh@CentOS7-openssl 1.0.x-x86-64' 
```

Now, each dataset represents a variant.

# References

- 📖 [**Ref1**] [A Multi-Armed Bandit Approach for Test Case Prioritization in Continuous Integration Environments](https://doi.org/10.1109/TSE.2020.2992428) published at **IEEE Transactions on Software Engineering (TSE)**
- 📖 [**Ref2**] [Learning-based prioritization of test cases in continuous integration of highly-configurable software](https://doi.org/10.1145/3382025.3414967) published at **Proceedings of the 24th ACM Conference on Systems and Software Product Line (SPLC'20)**

# Contributors

- 👨‍💻 Jackson Antonio do Prado Lima <a href="mailto:jacksonpradolima@gmail.com">:e-mail:</a>

